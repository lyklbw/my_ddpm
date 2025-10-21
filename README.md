# README

Code for 'A Conditional Denoising Diffusion Probabilistic Model for Radio Interferometic Image Reconstruction'.
Preprint: https://arxiv.org/abs/2305.09121 .


### Dataset

We use public dataset which is presented by Wu et al.[1]. Please find the dataset in https://github.com/wubenjamin/neural-interferometry .

Please download the data and modify the related path in the code.


### Testing

Please download the trained model from https://drive.google.com/drive/folders/12QelF9f_FJaR02Le81eSTfhC7kE4AgpR?usp=sharing
Then modify the "model_save_dir" and run testing.

```
SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset smos \
--batch_size 1 --num_workers 2"
TEST_FLAGS="--model_save_dir ./smos_model/ --resume_checkpoint ema_0.9999_003000.pt \
--output_dir ./test_output \
--debug_mode False"

torchrun --standalone --nproc_per_node=1 test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```


### Training

```
SCRIPT_FLAGS="--method_type vicddpm"
DATASET_FLAGS="--dataset smos --batch_size 1024 --num_workers 64"
TRAIN_FLAGS="--microbatch 1024 --save_interval 2000 --max_step 200000 --model_save_dir ./smos_model"

torchrun --standalone --nproc_per_node=2 train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS
```

/home/micdz/miniconda3/envs/ddpm_rfi/bin/torchrun --standalone --nproc_per_node=1 train.py \
  --method_type vicddpm \
  --dataset smos \
  --batch_size 16 \
  --num_workers 6 \
  --microbatch 32 \
  --save_interval 10000 \
  --max_step 150000 \
  --model_save_dir ./smos_model


Note
- The torchrun options must come before the script name. If you run `torchrun train.py --standalone ...`, or pass `--standalone`/`--nproc_per_node` to Python directly, you will get an error like: `train.py: error: unrecognized arguments: --standalone --nproc_per_node=1 ...`.
- Single-GPU alternatives:
	- Using torchrun: `torchrun --standalone --nproc_per_node=1 train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS`
	- Or without torchrun: `python train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS` (omit any torchrun flags)






### Reference

[1] Wu, Benjamin, et al. "Neural Interferometry: Image Reconstruction from Astronomical Interferometers using Transformer-Conditioned Neural Fields." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 36. No. 3. 2022.
