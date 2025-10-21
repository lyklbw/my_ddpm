SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset smos --batch_size 1 --num_workers 1"
TRAIN_FLAGS="--microbatch 1 --save_interval 10000 --max_step 1000 --model_save_dir ./smos_model"

torchrun --standalone --nproc_per_node=2 train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS