# 单机多卡
export TORCH_CUDA_ARCH_LIST="8.9"
deepspeed --include localhost:4,5,6,7 \
  train_rm.py \
  --model Qwen/Qwen3-1.7B \
  --train_data rlhf_pair.jsonl \
  --val_data rlhf_pair_val.jsonl \
  --output_dir ./output/qwen3_1_7b \
  --deepspeed_config ds_config.json \
  --dummy_batch_size 2 \
  --num_epochs 2 \
  --max_length 2048
