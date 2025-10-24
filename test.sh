SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset smos \
--batch_size 1 --num_workers 2"
TEST_FLAGS="--model_save_dir ./smos_model_10_22/ --resume_checkpoint ema_0.9999_050000.pt \
--output_dir ./test_output \
--debug_mode False"

torchrun --standalone --nproc_per_node=1 test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS