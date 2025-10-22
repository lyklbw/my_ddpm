torchrun --standalone --nproc_per_node=2 train.py \
  --method_type vicddpm \
  --dataset smos \
  --batch_size 824 \
  --num_workers 64 \
  --microbatch 824 \
  --save_interval 10_000 \
  --max_step 100_000 \
  --model_save_dir ./smos_model \
  --lr_anneal_steps 80_000