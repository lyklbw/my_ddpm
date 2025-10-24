torchrun --standalone --nproc_per_node=1 train.py \
  --method_type vicddpm \
  --dataset smos \
  --batch_size 820 \
  --num_workers 32 \
  --microbatch 820 \
  --save_interval 10_000 \
  --max_step 100_000 \
  --model_save_dir ./smos_model_10_22 \
  --lr_anneal_steps 80_000