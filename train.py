# In train.py
import argparse
import os
from utils import dist_util, logger
from utils.ddpm_utils.resample import create_named_schedule_sampler
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from utils.setting_utils import (
    dataset_setting, vicddpm_setting, unet_setting,
)
from utils.train_utils.vicddpm_train_util import VICDDPMTrainLoop
# from utils.train_utils.unet_train_util import UNetTrainLoop # Comment out if not used

def main():
    args = create_argparser().parse_args()

    # distributed setting (no changes needed)
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("making device configuration...")

    # Select method settings (no changes needed)
    if args.method_type == "vicddpm":
        method_setting = vicddpm_setting
    elif args.method_type == "unet":
        method_setting = unet_setting
    else:
        raise ValueError("Invalid method_type specified") # Added clarity

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)

    # create or load model (no changes needed for model creation itself)
    logger.log("creating model...")
    if args.resume_checkpoint:
        model_args = load_args_dict(os.path.join(args.model_save_dir, "model_args.pkl"))
    else:
        # Keys here should match vicddpm_setting.model_defaults()
        model_args = args_to_dict(args, method_setting.model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.model_save_dir, "model_args.pkl"))
    model = method_setting.create_model(**model_args)
    import ipdb; ipdb.set_trace()
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    if args.resume_checkpoint:
        data_args = load_args_dict(os.path.join(args.model_save_dir, "data_args.pkl"))
    else:
        # ---> Key Change: Ensure keys match updated dataset_setting.training_dataset_defaults() <---
        data_keys = dataset_setting.training_dataset_defaults().keys()
        logger.log(f"Extracting data args with keys: {list(data_keys)}") # Debug log
        data_args = args_to_dict(args, data_keys)
        save_args_dict(data_args, os.path.join(args.model_save_dir, "data_args.pkl"))
    # The create_training_dataset function now handles the 'smos' type correctly
    data = dataset_setting.create_training_dataset(**data_args)

    logger.log("training...")
    # Training args loading (no changes needed)
    if args.resume_checkpoint:
        training_args = load_args_dict(os.path.join(args.model_save_dir, "training_args.pkl"))
        training_args["resume_checkpoint"] = args.resume_checkpoint
    else:
        # Keys here should match vicddpm_setting.training_setting_defaults()
        training_args = args_to_dict(args, method_setting.training_setting_defaults().keys())
        save_args_dict(training_args, os.path.join(args.model_save_dir, "training_args.pkl"))

    # Diffusion and Sampler creation (no changes needed)
    if args.method_type == "vicddpm":
        logger.log("creating diffusion...")
        if args.resume_checkpoint:
            diffusion_args = load_args_dict(os.path.join(args.model_save_dir, "diffusion_args.pkl"))
        else:
            diffusion_args = args_to_dict(args, method_setting.diffusion_defaults().keys())
            save_args_dict(diffusion_args, os.path.join(args.model_save_dir, "diffusion_args.pkl"))
        diffusion = method_setting.create_gaussian_diffusion(**diffusion_args)

        logger.log("creating schedule_sampler...")
        if args.resume_checkpoint:
            schedule_sampler_args = load_args_dict(os.path.join(args.model_save_dir, "schedule_sampler_args.pkl"))
        else:
            schedule_sampler_args = args_to_dict(args, method_setting.schedule_sampler_setting_defaults().keys())
            save_args_dict(schedule_sampler_args, os.path.join(args.model_save_dir, "schedule_sampler_args.pkl"))
        schedule_sampler = create_named_schedule_sampler(**schedule_sampler_args, diffusion=diffusion)

        # Run training loop (no changes needed here)
        VICDDPMTrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            schedule_sampler=schedule_sampler,
            **training_args,
        ).run_loop()

    elif args.method_type == "unet":
        # Keep UNetTrainLoop if needed, otherwise remove
        # UNetTrainLoop(
        #     model=model,
        #     data=data,
        #     **training_args,
        # ).run_loop()
        logger.log("UNet method type selected, but loop might be specific to VICDDPM data handling now.")
        pass # Or raise error if UNet is no longer supported with SMOS data loader

    logger.log("complete training.\n")

def create_argparser():
    # Base defaults (no changes needed)
    defaults = dict(
        method_type="vicddpm",
        log_dir="logs",
        local_rank=0,
    )
    # ---> Key Change: Update with the *modified* dataset defaults first <---
    defaults.update(dataset_setting.training_dataset_defaults())

    # Update with method-specific defaults (this part remains dynamic)
    # We parse method_type first to load the correct subsequent defaults
    parser_temp = argparse.ArgumentParser()
    add_dict_to_argparser(parser_temp, defaults) # Add base and DATASET defaults
    args_temp, _ = parser_temp.parse_known_args() # Parse known args to get method_type

    if args_temp.method_type == "vicddpm":
        defaults.update(vicddpm_setting.model_defaults())
        defaults.update(vicddpm_setting.diffusion_defaults())
        defaults.update(vicddpm_setting.training_setting_defaults())
        defaults.update(vicddpm_setting.schedule_sampler_setting_defaults())
        # dataset defaults already added
    elif args_temp.method_type == "unet":
        defaults.update(unet_setting.model_defaults())
        defaults.update(unet_setting.training_setting_defaults())
        # dataset defaults already added
    else:
         raise ValueError(f"Unknown method_type: {args_temp.method_type}")

    # Create the final parser with all defaults
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()