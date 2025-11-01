#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import argparse
import json
import time
from typing import Any, Dict, List

import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.inference_mode()
def evaluate(engine, val_loader, log_step: int, epoch: int):
    """只在 rank0 调用"""
    engine.module.eval() if hasattr(engine, "module") else engine.eval()
    total_loss = 0.0
    total_cnt = 0
    correct = 0

    for batch in tqdm(val_loader, desc="Evaluating"):
        pos, neg, adv = batch
        pos = {k: v.to(engine.device) for k, v in pos.items()}
        neg = {k: v.to(engine.device) for k, v in neg.items()}
        adv = adv.to(engine.device)

        r_pos = engine(**pos).logits.squeeze(-1)  # [B]
        r_neg = engine(**neg).logits.squeeze(-1)  # [B]
        s = r_pos - r_neg  # [B]
        loss = loss_fn(r_pos, r_neg, adv)  # [B]

        total_loss += float(loss.detach().cpu()) * pos["input_ids"].size(0)
        total_cnt += pos["input_ids"].size(0)
        correct += int((s > 0).sum().item())

    avg_loss = total_loss / max(1, total_cnt)
    acc = correct / max(1, total_cnt)

    # 记录到 SwanLab（rank0）
    swanlab.log({"eval/loss": avg_loss, "eval/acc": acc, "eval/epoch": epoch}, step=log_step)

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
        ex = self.samples[idx]
        return ex


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
            # 保持原有行为：从样本读取 adv 字段
            adv_list.append(ex["advantage"])

        pos_toks = self.tok(pos_txts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True)
        neg_toks = self.tok(neg_txts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True)
        adv = torch.tensor(adv_list, dtype=torch.float)

        return pos_toks, neg_toks, adv


def loss_fn(r_pos, r_neg, adv):
    s = r_pos - r_neg  # [B]
    loss = -F.logsigmoid(s) * adv  # [B]
    return loss.mean()


# -------- 训练循环 ----------
def train(args):
    deepspeed.init_distributed()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

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
    # DataLoader 的 batch size/accumulation 由 DeepSpeed 配置控制；此处给个占位不冲突
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=args.dummy_batch_size,  # 实际以 deepspeed json 为准
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=collate,
    #     sampler=DistributedSampler(dataset),
    # )
    if args.val_data is not None and deepspeed.comm.get_rank() == 0:
        val_dataset = PairwiseJSONLDataset(args.val_data)
        val_collate = PairwiseCollator(tokenizer, max_length=args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,  # 实际无关 DS 配置，只是 rank0 上本地跑评估
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
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
    if deepspeed.comm.get_rank() == 0:
        train_loader = tqdm(train_loader, desc="Training")
        swanlab.init(
            project=args.swan_project,
            run_name=args.swan_run,
            config=args,
        )

    engine.train()
    global_step = 1
    start_time = time.time()
    for epoch in range(1, 1 + args.num_epochs):
        loss_epoch_mean, r_pos_epoch_mean, r_neg_epoch_mean = 0.0, 0.0, 0.0
        for pos, neg, adv in train_loader:
            # 将 batch 张量放到当前设备
            pos = {k: v.to(engine.device) for k, v in pos.items()}
            neg = {k: v.to(engine.device) for k, v in neg.items()}
            adv = adv.to(engine.device)
            # engine 就是模型，可直接调用
            r_pos = engine(**pos).logits.squeeze(-1)  # [B]
            r_neg = engine(**neg).logits.squeeze(-1)  # [B]
            loss = loss_fn(r_pos, r_neg, adv)  # [B]
            engine.backward(loss)
            engine.step()

            loss_epoch_mean += float(loss.detach().mean().cpu())
            r_pos_epoch_mean += float(r_pos.detach().mean().cpu())
            r_neg_epoch_mean += float(r_neg.detach().mean().cpu())
            global_step += 1

            if deepspeed.comm.get_rank() == 0 and global_step % args.log_every == 0:
                swanlab.log(
                    {
                        "train/loss_microbatch": loss_epoch_mean / args.log_every,
                        "train/r_pos_microbatch": r_pos_epoch_mean / args.log_every,
                        "train/r_neg_microbatch": r_neg_epoch_mean / args.log_every,
                        "train/adv_microbatch": (r_pos_epoch_mean - r_neg_epoch_mean) / args.log_every,
                    },
                    step=global_step // args.log_every,
                )
                loss_epoch_mean = r_pos_epoch_mean = r_neg_epoch_mean = 0.0

        # 每个 epoch 保存一次（rank0）
        if deepspeed.comm.get_rank() == 0:
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
                json.dump(
                    {"num_labels": 1, "arch": "AutoModelForSequenceClassification"}, f, ensure_ascii=False, indent=2
                )

        # 每隔若干个 epoch 做一次评估（只在 rank0）
        if (epoch) % args.eval_every_epochs == 0:
            torch.distributed.barrier()
            if val_loader and deepspeed.comm.get_rank() == 0:
                avg_loss, acc = evaluate(engine, val_loader, global_step // args.log_every, epoch)
                print(f"[Eval @ epoch {epoch}] loss={avg_loss:.4f} acc={acc:.4f}")
            torch.distributed.barrier()

    if deepspeed.comm.get_rank() == 0:
        swanlab.log({"time/total_sec": time.time() - start_time})
        swanlab.finish()


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--train_data", type=str, required=True)  # JSONL
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--deepspeed_config", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dummy_batch_size", type=int, default=1)
    # SwanLab
    p.add_argument("--swan_project", type=str, default="reward-model")
    p.add_argument("--swan_run", type=str, default=f"rm-training_{int(time.time())}")
    # build_args() 里追加
    p.add_argument("--val_data", type=str, default=None)  # JSONL，与训练集同格式
    p.add_argument("--eval_every_epochs", type=int, default=1)  # 每隔多少个 epoch 做一次评估
    p.add_argument("--val_batch_size", type=int, default=1)  # 仅用于 DataLoader，占位即可
    p.add_argument("--local_rank", type=int)
    p.add_argument("--log_every", type=int, default=192)

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
