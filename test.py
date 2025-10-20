# In test.py
import argparse
import os
from utils import dist_util, logger
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict # Removed save_args_dict
from utils.setting_utils import (
    dataset_setting, vicddpm_setting, unet_setting,
)
from utils.test_utils.vicddpm_test_util import VICDDPMTestLoop
# from utils.test_utils.unet_test_util import UNetTestLoop # Comment out if not used

def main():
    args = create_argparser().parse_args()

    # distributed setting (no changes needed)
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True) # Usually write=False for testing logs?
    logger.log("making device configuration...")

    # Select method settings (no changes needed)
    if args.method_type == "vicddpm":
        method_setting = vicddpm_setting
    elif args.method_type == "unet":
        method_setting = unet_setting
    else:
        raise ValueError("Invalid method_type specified")

    # load model args from training save dir (no changes needed)
    logger.log("creating model...")
    model_args_path = os.path.join(args.model_save_dir, "model_args.pkl")
    if not os.path.exists(model_args_path):
         raise FileNotFoundError(f"Model args file not found: {model_args_path}. Ensure --model_save_dir is correct.")
    model_args = load_args_dict(model_args_path)
    model = method_setting.create_model(**model_args)
    model.to(dist_util.dev())
    # Note: Model loading happens inside the TestLoop class, using args.resume_checkpoint

    logger.log("creating data loader...")
    # ---> Key Change: Ensure keys match updated dataset_setting.test_dataset_defaults() <---
    data_keys = dataset_setting.test_dataset_defaults().keys()
    logger.log(f"Extracting data args with keys: {list(data_keys)}") # Debug log
    data_args = args_to_dict(args, data_keys)
    # The create_test_dataset function now handles the 'smos' type correctly
    data = dataset_setting.create_test_dataset(**data_args)

    logger.log("test...")
    # ---> Key Change: Ensure keys match method's test_setting_defaults <---
    test_keys = method_setting.test_setting_defaults().keys()
    logger.log(f"Extracting test args with keys: {list(test_keys)}") # Debug log
    test_args = args_to_dict(args, test_keys)


    if args.method_type == "vicddpm":
        logger.log("creating diffusion...")
        # ---> Key Change: Need diffusion settings from training save dir OR from defaults/cmd line <---
        # Option 1: Load from saved args (consistent with training)
        diffusion_args_path = os.path.join(args.model_save_dir, "diffusion_args.pkl")
        if os.path.exists(diffusion_args_path):
             diffusion_args = load_args_dict(diffusion_args_path)
             logger.log("Loaded diffusion args from saved file.")
        else:
             # Option 2: Get from command line/defaults (might differ from training)
             logger.log("WARN: diffusion_args.pkl not found. Using defaults/command line args for diffusion.")
             diffusion_args = args_to_dict(args, method_setting.diffusion_defaults().keys())

        diffusion = method_setting.create_gaussian_diffusion(**diffusion_args)

        # Ensure image_size is passed correctly to the test loop if needed
        if 'image_size' not in test_args:
             test_args['image_size'] = model_args.get('image_size', args.image_size) # Get from loaded model args or cmd line
             logger.log(f"Added image_size={test_args['image_size']} to test_args")

        # Pass necessary args to Test Loop
        VICDDPMTestLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            **test_args, # Contains batch_size, model_save_dir, resume_checkpoint etc.
        ).run_loop()

    elif args.method_type == "unet":
         # Keep UNetTestLoop if needed, otherwise remove
        # UNetTestLoop(
        #     model=model,
        #     data=data,
        #     **test_args,
        # ).run_loop()
        logger.log("UNet method type selected, but loop might be specific to VICDDPM data handling now.")
        pass # Or raise error

    logger.log("complete test.\n")

def create_argparser():
    # Base defaults (no changes needed)
    defaults = dict(
        method_type="vicddpm",
        log_dir="logs_test", # Different log dir for testing
        local_rank=0,
    )
    # ---> Key Change: Update with the *modified* dataset TEST defaults first <---
    defaults.update(dataset_setting.test_dataset_defaults())

    # Update with method-specific defaults (this part remains dynamic)
    parser_temp = argparse.ArgumentParser()
    add_dict_to_argparser(parser_temp, defaults) # Add base and DATASET defaults
    args_temp, _ = parser_temp.parse_known_args() # Parse known args to get method_type

    if args_temp.method_type == "vicddpm":
        # Need model defaults ONLY for things potentially NOT saved in model_args.pkl (like image_size if padding changed)
        # OR things needed by diffusion creation if not loading saved diffusion args
        defaults.update(vicddpm_setting.model_defaults()) # Include for safety, e.g., image_size default
        defaults.update(vicddpm_setting.diffusion_defaults()) # Needed if diffusion_args.pkl isn't found
        defaults.update(vicddpm_setting.test_setting_defaults()) # Add test-specific args
        # Ensure test defaults override training defaults if names conflict (e.g., batch_size)
    elif args_temp.method_type == "unet":
        defaults.update(unet_setting.model_defaults())
        defaults.update(unet_setting.test_setting_defaults())
        # dataset defaults already added
    else:
        raise ValueError(f"Unknown method_type: {args_temp.method_type}")

    # Create the final parser with all defaults
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()