import matplotlib.pyplot as plt

from utils.script_util import save_args_dict, load_args_dict
from utils.test_utils.base_test_util import *
from utils.galaxy_data_utils.transform_util import *


MAX_NUM_SAVED_SAMPLES = 5

test_num = 0
class DDPMTestLoop(TestLoop):

    def __init__(self, diffusion, image_size, num_samples_per_mask=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.image_size = image_size
        self.num_samples_per_mask = num_samples_per_mask


    def run_loop(self):
        super().run_loop()

    def forward_backward(self, data_item):
        global test_num
        img, batch_kwargs = data_item
        file_name = batch_kwargs["file_name"]
        slice_index = test_num
        test_num = test_num+1
        samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        if os.path.exists(samples_path):
            if os.path.exists(os.path.join(samples_path, "slice_information.pkl")):
                logger.log(f"have sampled for {file_name} slice {slice_index}")
                return
        else:
            os.makedirs(samples_path, exist_ok=True)

        k_samples = self.sample(batch_kwargs)
        # self.save_samples(k_samples, samples_path, batch_kwargs)
        logger.log(f"complete sampling for {file_name} slice {slice_index}")

    def sample(self, batch_kwargs):
        """
        The sample process is defined in children class.
        """
        pass

    # Example modification/override in utils/test_utils/vicddpm_test_util.py
# Inside class VICDDPMTestLoop(DDPMTestLoop):

    def save_samples(self, generated_samples, samples_path, batch_kwargs):
        # generated_samples shape: [num_samples, 2, 69, 69] (if cropped)
        # batch_kwargs contains original 'target' and 'condition' (padded)

        # --- Optional: Crop original target and condition for comparison ---
        target_padded = batch_kwargs['target'][0].cpu().numpy() # Get first item in batch [2, 80, 80]
        condition_padded = batch_kwargs['condition'][0].cpu().numpy() # [2, 80, 80]

        h_orig, w_orig = 69, 69
        h_padded, w_padded = self.image_size, self.image_size
        pad_h = h_padded - h_orig
        pad_w = w_padded - w_orig
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        target_cropped = target_padded[ :, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig] # [2, 69, 69]
        condition_cropped = condition_padded[:, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig] # [2, 69, 69]
        # --- End Optional Cropping ---

        # Save cropped versions as .npy
        np.save(os.path.join(samples_path, "target_original.npy"), target_cropped)
        np.save(os.path.join(samples_path, "input_corrupted.npy"), condition_cropped)

        # Save the mean of the generated samples
        mean_sample = np.mean(generated_samples, axis=0) # Shape [2, 69, 69]
        np.save(os.path.join(samples_path, "generated_mean.npy"), mean_sample)

        # Save individual samples if needed
        for i in range(min(MAX_NUM_SAVED_SAMPLES, len(generated_samples))):
             np.save(os.path.join(samples_path, f"generated_sample_{i + 1}.npy"), generated_samples[i])

        # Optional: Save visualization of magnitude (channel 0 assumed real, 1 imag)
        try:
            target_mag = np.sqrt(target_cropped[0]**2 + target_cropped[1]**2)
            condition_mag = np.sqrt(condition_cropped[0]**2 + condition_cropped[1]**2)
            mean_sample_mag = np.sqrt(mean_sample[0]**2 + mean_sample[1]**2)

            plt.imsave(fname=os.path.join(samples_path, "target_original_mag.png"), arr=target_mag, cmap="viridis") # Use viridis or gray
            plt.imsave(fname=os.path.join(samples_path, "input_corrupted_mag.png"), arr=condition_mag, cmap="viridis")
            plt.imsave(fname=os.path.join(samples_path, "generated_mean_mag.png"), arr=mean_sample_mag, cmap="viridis")
        except Exception as e:
            print(f"Could not save magnitude images: {e}")

        # Remove or adapt saving of slice_information.pkl if not relevant
        # save_args_dict(...)

        image = batch_kwargs["image"][0][0]
        image_dir = batch_kwargs["image_dir"][0][0]


        for to_save_image, name in zip([image, image_dir], ["image", "image_dir"]):
            plt.imsave(
                fname=os.path.join(samples_path, f"{name}.png"),
                arr=to_save_image, cmap="hot"
            )

        # save some samples, less than 5
        # for i in range(min(MAX_NUM_SAVED_SAMPLES, len(samples))):
        #     sample = np.abs(samples[i][0])
        #     plt.imsave(
        #         fname=os.path.join(samples_path, f"sample_{i + 1}.png"),
        #         arr=sample, cmap="hot"
        #     )

        mean_sample = np.mean(k_samples, axis=0)
        np.save(os.path.join(samples_path, "asample_mean.npy"), mean_sample)
        plt.imsave(fname=os.path.join(samples_path, "asample_mean.png"), arr=mean_sample[0], cmap="hot")
        # save all information
        # np.savez(os.path.join(samples_path, f"all_samples"), samples0)  # arr is not magnitude images
        # saved_args = {
        #     "scale_coeff": batch_kwargs["scale_coeff"],
        #     "slice_index": batch_kwargs["slice_index"],
        #     "image": batch_kwargs["image"][0:1, ...],
        # }
        # save_args_dict(saved_args, os.path.join(samples_path, "slice_information.pkl"))


def extract_slice_index(slice_index):
    return int(slice_index.split("_")[-1])
