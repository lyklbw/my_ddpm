# In utils/test_utils/vicddpm_test_util.py
from utils.test_utils.ddpm_test_util import DDPMTestLoop, MAX_NUM_SAVED_SAMPLES, test_num # Import test_num if used by parent/original logic
from utils import dist_util, logger # Added logger import
import matplotlib.pyplot as plt
from utils.galaxy_data_utils.transform_util import * # Keep th2np_magnitude if used
import numpy as np
import torch
import os # Import os

class VICDDPMTestLoop(DDPMTestLoop):

    def __init__(self, *args, **kwargs):
        # Note: image_size should be passed from test_args (e.g., 80)
        super().__init__(*args, **kwargs)
        # self.uv_dense = np.load("./data/uv_dense.npy") # Removed
        # self.uv_dense = torch.tensor(self.uv_dense) # Removed
        print(f"VICDDPMTestLoop initialized with image_size: {self.image_size}")
        # Initialize a list to store MAE values for averaging later
        self.mae_list = []


    # <<< --- Overridden forward_backward method --- >>>
    def forward_backward(self, data_item):
        """
        Overrides the parent method to handle dictionary data_item and calculate MAE.
        """
        global test_num # Keep global counter if used by parent/original logic
        # data_item is {'target': tensor, 'condition': tensor}
        batch_kwargs = data_item

        # file_name = f"item_{test_num}" # Use counter directly
        slice_index = test_num # Use counter as slice index
        test_num = test_num + 1

        # samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        # os.makedirs(samples_path, exist_ok=True) # Ensure directory exists

        # logger.log(f"Sampling for {file_name} slice {slice_index}...")
        cropped_generated_samples = self.sample(batch_kwargs) # Gets [num_samples, 2, 69, 69]
        # logger.log(f"Saving samples and calculating MAE for {file_name} slice {slice_index}...")

        # Calculate MAE before saving (or within save_samples)
        # Extract target tensor (ground truth)
        target_padded = batch_kwargs['target'][0].cpu().numpy() # Get first item in batch [2, 80, 80]

        # Crop target tensor
        h_orig, w_orig = 69, 69
        h_padded, w_padded = self.image_size, self.image_size
        pad_h = h_padded - h_orig
        pad_w = w_padded - w_orig
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        target_cropped = target_padded[ :, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig] # [2, 69, 69]

        # Calculate mean of generated samples
        mean_generated_cropped = np.mean(cropped_generated_samples, axis=0) # Shape [2, 69, 69]
        # mean_generated_cropped = cropped_generated_samples[-1]
        # Calculate Mean Absolute Error (MAE)
        # Compares the mean generated image with the ground truth
        absolute_diff = np.abs(mean_generated_cropped - target_cropped)
        mae = np.mean(absolute_diff)
        print(f"MAE is : {mae}")
        self.mae_list.append(mae) # Store MAE for averaging later

        
        

        # # Save the results (including the calculated mean sample)
        # self.save_samples(cropped_generated_samples, mean_generated_cropped, target_cropped, samples_path, batch_kwargs)

        # logger.log(f"Completed processing for {file_name} slice {slice_index}")

    # <<< --- run_loop override to calculate average MAE --- >>>
    def run_loop(self):
        super().run_loop() # Call the parent run_loop to iterate through data

        # After processing all samples, calculate and log the average MAE
        if self.mae_list:
            average_mae = np.mean(self.mae_list)
            logger.log(f"\n--------------------------------------------------")
            logger.log(f"Average MAE over {len(self.mae_list)} samples: {average_mae:.6f}")
            logger.log(f"--------------------------------------------------")
            # Log average MAE to tensorboard/kv dict
            logger.log_kv("MAE_average", average_mae)
            # Make sure to write the final averages if using logger's kv system
            # logger.write_kv(self.step) # 'step' might not be the right index here, maybe len(self.mae_list)?
        else:
            logger.log("No MAE values were calculated.")

    def sample(self, batch_kwargs):
        import ipdb; ipdb.set_trace()
        # ... (Your existing sample method code, which already returns cropped_samples) ...
        condition_tensor = batch_kwargs['condition'].to(dist_util.dev())
        if condition_tensor.shape[0] == 1 and self.batch_size > 1:
            condition_tensor = condition_tensor.repeat(self.batch_size, 1, 1, 1)
        cond = {'condition_input': condition_tensor}
        
        samples = []
        
        while len(samples) * self.batch_size < self.num_samples_per_mask:
            sample_shape = (self.batch_size, 2, self.image_size, self.image_size)
            # print(f"Sampling with shape: {sample_shape}") # Debug
            sample, samples_denoise = self.diffusion.sample_loop(
                self.model,
                sample_shape,
                cond,
                clip=False
            )



            samples.append(sample.cpu().detach().numpy())
            break

        samples = np.concatenate(samples, axis=0)
        # samples = samples[: self.num_samples_per_mask]

        # plot samples every 100 indices
        import matplotlib.pyplot as plt
        import ipdb; ipdb.set_trace()

        for i in range(len(samples_denoise)):
            if (i % 111 != 0) :
                continue 
            plt.imsave(fname=os.path.join(self.output_dir, f"sample_{i}_mag.png"),
                       arr=np.sqrt(samples_denoise[i][0][0] ** 2 + samples_denoise[i][0][1] ** 2), cmap="viridis")
            print(f"Saved sample_{i}_mag.png to {self.output_dir}")
        
            

        h_orig, w_orig = 69, 69
        h_padded, w_padded = self.image_size, self.image_size
        pad_h = h_padded - h_orig
        pad_w = w_padded - w_orig
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        cropped_samples = samples[:, :, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig]
        # print(f"Cropped samples shape: {cropped_samples.shape}") # Debug
        return cropped_samples


    # <<< --- Modified save_samples to accept pre-calculated values --- >>>
    def save_samples(self, generated_samples_cropped, mean_generated_cropped, target_cropped, samples_path, batch_kwargs):
         # generated_samples_cropped shape: [num_samples, 2, 69, 69]
         # mean_generated_cropped shape: [2, 69, 69]
         # target_cropped shape: [2, 69, 69]
         # batch_kwargs contains original 'condition' (padded)

        # Crop condition tensor
        condition_padded = batch_kwargs['condition'][0].cpu().numpy() # [2, 80, 80]
        h_orig, w_orig = 69, 69
        h_padded, w_padded = self.image_size, self.image_size
        pad_h = h_padded - h_orig
        pad_w = w_padded - w_orig
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        condition_cropped = condition_padded[:, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig] # [2, 69, 69]

        # Save cropped versions as .npy
        # np.save(os.path.join(samples_path, "target_original.npy"), target_cropped)
        # np.save(os.path.join(samples_path, "input_corrupted.npy"), condition_cropped)
        # np.save(os.path.join(samples_path, "generated_mean.npy"), mean_generated_cropped)

        # Save individual samples if needed
        # for i in range(min(MAX_NUM_SAVED_SAMPLES, len(generated_samples_cropped))):
        #      np.save(os.path.join(samples_path, f"generated_sample_{i + 1}.npy"), generated_samples_cropped[i])

        # # Optional: Save visualization of magnitude
        # try:
        #     target_mag = np.sqrt(target_cropped[0]**2 + target_cropped[1]**2)
        #     condition_mag = np.sqrt(condition_cropped[0]**2 + condition_cropped[1]**2)
        #     mean_sample_mag = np.sqrt(mean_generated_cropped[0]**2 + mean_generated_cropped[1]**2)

        #     plt.imsave(fname=os.path.join(samples_path, "target_original_mag.png"), arr=target_mag, cmap="viridis")
        #     plt.imsave(fname=os.path.join(samples_path, "input_corrupted_mag.png"), arr=condition_mag, cmap="viridis")
        #     plt.imsave(fname=os.path.join(samples_path, "generated_mean_mag.png"), arr=mean_sample_mag, cmap="viridis")
        # except Exception as e:
        #     print(f"Could not save magnitude images: {e}")