#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import time
from typing import Any, Dict, List

import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.inference_mode()
def evaluate(engine, val_loader, log_step: int, args: bool):
    """只在 rank0 调用"""
    engine.module.eval() if hasattr(engine, "module") else engine.eval()
    total_loss = 0.0
    total_cnt = 0
    correct = 0

    for prompt_toks, adv in tqdm(val_loader, desc="Evaluating"):
        bs = prompt_toks["input_ids"].size(0)
        prompt_toks = {k: v.to(engine.device, non_blocking=True) for k, v in prompt_toks.items()}
        adv = adv.to(engine.device, non_blocking=True)

        reward = engine(**prompt_toks).logits.squeeze(-1)  # [B]
        r_pos, r_neg = reward.chunk(2, dim=0)  # [B/2], [B/2]
        s = r_pos - r_neg  # [B]
        loss = loss_fn(r_pos, r_neg, adv, args)  # [B]

        total_loss += float(loss.detach().cpu()) * bs
        total_cnt += bs
        correct += (s > 0).sum().item()

    avg_loss = total_loss / max(1, total_cnt)
    acc = correct / max(1, total_cnt)

    # 记录到 SwanLab（rank0）
    swanlab.log({"eval/loss": avg_loss, "eval/acc": acc}, step=log_step)

    # 切回训练模式
    engine.module.train() if hasattr(engine, "module") else engine.train()

    return avg_loss, acc


# -------- 数据集 ----------
class PairwiseJSONLDataset(Dataset):
    """
    JSONL 每行支持：
    {
      "question": "...",
      "chosen": "...",
      "rejected": "...",
      "advantage": 1.7
    }
    """

    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                self.samples.append(ex)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------- Collate ----------
class PairwiseCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048, add_eos: bool = True):
        self.tok = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos

    def apply_template(self, messages) -> str:
        # 使用 tokenizer.apply_chat_template 构造聊天格式文本，替代手动拼接
        # messages 包括 user（问题）和 assistant（回复）两部分，适用于打分场景
        # 这里我们已经提供了完整的回复，不需要额外的 generation prompt
        text = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return text

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos_txts = []
        neg_txts = []
        adv_list = []
        for ex in batch:
            message_pos = [
                {"system": "You are a helpful assistant."},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["chosen"]},
            ]
            message_neg = [
                {"system": "You are a helpful assistant."},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["rejected"]},
            ]
            pos_txts.append(self.apply_template(message_pos))
            neg_txts.append(self.apply_template(message_neg))
            adv_list.append(ex["advantage"])

        prompt_toks = self.tok(
            pos_txts + neg_txts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True
        )
        adv = torch.tensor(adv_list, dtype=torch.float)

        return prompt_toks, adv


def loss_fn(r_pos, r_neg, adv, args):
    s = r_pos - r_neg  # [B]

    if args.use_advantage:
        s *= adv

    loss = -F.logsigmoid(s)

    if args.use_reward_reg:
        # 奖励正则化，鼓励奖励值接近 0
        loss += args.reward_reg_alpha * ((r_pos + r_neg) ** 2)

    return loss.mean()


# -------- 训练循环 ----------
def train(args):
    deepspeed.init_distributed()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    rank = deepspeed.comm.get_rank()

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="right")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False

    # Dataset / Loader
    dataset = PairwiseJSONLDataset(args.train_data)
    collate = PairwiseCollator(tokenizer, max_length=args.max_length)

    if args.val_data and rank == 0:
        val_dataset = PairwiseJSONLDataset(args.val_data)
        val_collate = PairwiseCollator(tokenizer, max_length=args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,  # 无关 DS 配置，只是 rank0 上本地跑评估
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            collate_fn=val_collate,
        )

    # DeepSpeed 初始化（优化器/调度器等从 ds_config.json 里拿）
    ds_config = json.load(open(args.deepspeed_config))
    engine, _, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=dataset,
        collate_fn=collate,
    )
    # SwanLab
    if rank == 0:
        swanlab.init(
            project=args.swan_project,
            run_name=args.swan_run,
            config=args,
        )

    engine.train()
    global_step = 1
    start_time = time.time()
    loss_mb, r_adv_mb, correct, all_cnt = 0, 0, 0, 0
    for epoch in range(1, 1 + args.num_epochs):
        for prompt_toks, adv in tqdm(train_loader, disable=rank != 0, smoothing=0.01):
            prompt_toks = {k: v.to(engine.device, non_blocking=True) for k, v in prompt_toks.items()}
            adv = adv.to(engine.device, non_blocking=True)
            bs = adv.size(0)

            reward = engine(**prompt_toks).logits.squeeze(-1)  # [B]
            r_pos, r_neg = reward.chunk(2, dim=0)  # [B/2], [B/2]
            loss = loss_fn(r_pos, r_neg, adv, args)  # [B]
            engine.backward(loss)
            engine.step()

            loss_mb += loss.detach().cpu().item()
            relative_score = r_pos.detach() - r_neg.detach()
            r_adv_mb += relative_score.sum().cpu().item()
            correct += (relative_score > 0).sum().item()
            all_cnt += bs
            global_step += 1

            if rank == 0 and global_step % args.log_every == 0:
                swanlab.log(
                    {
                        "train/loss_microbatch": loss_mb * bs / all_cnt,  # 这里乘 bs 是因为 loss 已经是均值了
                        "train/adv_microbatch": r_adv_mb / all_cnt,
                        "train/acc_microbatch": correct / all_cnt,
                    },
                    step=global_step // args.log_every,
                )
                loss_mb, r_adv_mb, correct, all_cnt = 0, 0, 0, 0

    # 每个 epoch 保存一次（rank0）
    if rank == 0:
        tag = f"epoch-{epoch}"
        save_dir = os.path.join(args.output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        # 按 16-bit/FP32 保存权重（依据 DeepSpeed 当前精度）
        engine.save_checkpoint(args.output_dir, tag=tag)
        # 额外保存 tokenizer（和一个便于 from_pretrained 的软链接）
        tokenizer.save_pretrained(save_dir)
        # 保存一个轻量 config，提示 num_labels=1
        cfg_path = os.path.join(save_dir, "rm_note.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"num_labels": 1, "arch": "AutoModelForSequenceClassification"}, f, ensure_ascii=False, indent=2)

    # torch.cuda.synchronize()
    # torch.distributed.barrier()

    # 每隔若干个 epoch 做一次评估（只在 rank0）
    if rank == 0:
        avg_loss, acc = evaluate(engine, val_loader, global_step // args.log_every, args)
        print(f"[Eval @ epoch {epoch}] loss={avg_loss:.4f} acc={acc:.4f}")

    if rank == 0:
        swanlab.log({"time/total_sec": time.time() - start_time})
        swanlab.finish()


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--deepspeed_config", type=str, required=True)
    p.add_argument("--train_data", type=str, required=True)  # JSONL
    p.add_argument("--val_data", type=str, default=None)  # JSONL，与训练集同格式
    p.add_argument("--val_batch_size", type=int, default=1)  # 仅用于 DataLoader，占位即可
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--eval_every_epochs", type=int, default=1)  # 每隔多少个 epoch 做一次评估
    p.add_argument("--log_every", type=int, default=32)
    p.add_argument("--use_advantage", action="store_true", default=False)
    p.add_argument("--use_reward_reg", action="store_true", default=False)
    p.add_argument("--reward_reg_alpha", type=float, default=0.01)
    # SwanLab
    p.add_argument("--swan_project", type=str, default="reward-model")
    p.add_argument("--swan_run", type=str, default=f"rm-training_{int(time.time())}")

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
