# In utils/setting_utils/dataset_setting.py
from utils.dataset_utils import galaxy, smos_dataset # 确保导入 smos_dataset

def training_dataset_defaults():
    """
    Defaults for training datasets. Updated for SMOS.
    """
    return dict(
        dataset="smos", # Changed default dataset
        # data_dir="../datasets/galaxy/knee_singlecoil_train", # Removed or commented out
        # data_info_list_path="", # Removed or commented out
        data_path_corrupted="./data/D_tensor_all.pt", # Added - Specify via command line
        data_path_original="./data/D_original_tensor_all.pt", # Added - Specify via command line
        batch_size=1,
        # acceleration=4, # Removed or commented out
        # random_flip=False, # Removed or commented out (can be added back to smos_dataset if needed)
        val_split_ratio=0.1, # Added default validation split ratio
        is_distributed=True,
        num_workers=0,
        seed=42 # Added seed for reproducibility
    )

def create_training_dataset(
        dataset,
        # data_dir, # Removed
        # data_info_list_path, # Removed
        data_path_corrupted, # Added
        data_path_original,  # Added
        batch_size,
        # acceleration, # Removed
        # random_flip, # Removed
        val_split_ratio, # Added
        is_distributed,
        num_workers,
        seed, # Added
        # post_process=None, # Removed or handled internally if needed
):
    if dataset == "smos":
        load_data = smos_dataset.load_smos_data
        return load_data(
            data_path_corrupted=data_path_corrupted,
            data_path_original=data_path_original,
            batch_size=batch_size,
            val_split_ratio=val_split_ratio,
            is_distributed=is_distributed,
            is_train=True,
            num_workers=num_workers,
            seed=seed,
        )
    elif dataset == "galaxy":
        # Keep original galaxy loading logic if needed
        load_data = galaxy.load_data
        return load_data(
            # Pass original galaxy args here...
            data_dir=data_dir, # You would need to add data_dir back if using galaxy
            data_info_list_path=data_info_list_path, # Add back if using galaxy
            batch_size=batch_size,
            random_flip=random_flip, # Add back if using galaxy
            is_distributed=is_distributed,
            is_train=True,
            post_process=None,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def test_dataset_defaults():
    """
    Defaults for test datasets. Updated for SMOS.
    """
    return dict(
        dataset="smos", # Changed default dataset
        # data_dir="../dataset/galaxy/knee_singlecoil_val", # Removed or commented out
        # data_info_list_path="", # Removed or commented out
        data_path_corrupted="", # Added - Specify via command line
        data_path_original="", # Added - Specify via command line
        batch_size=1, # Test batch size usually 1
        # acceleration=4, # Removed or commented out
        # random_flip=False, # Removed or commented out
        val_split_ratio=0.1, # Keep consistent, used to select the val split
        is_distributed=True,
        num_workers=0,
        seed=42 # Added seed for reproducibility
    )

def create_test_dataset(
        dataset,
        # data_dir, # Removed
        # data_info_list_path, # Removed
        data_path_corrupted, # Added
        data_path_original,  # Added
        batch_size,
        # acceleration, # Removed
        # random_flip, # Removed
        val_split_ratio, # Added
        is_distributed,
        num_workers,
        seed, # Added
        # post_process=None, # Removed or handled internally if needed
):
    # This function is very similar to create_training_dataset, just sets is_train=False
    if dataset == "smos":
        load_data = smos_dataset.load_smos_data
        return load_data(
            data_path_corrupted=data_path_corrupted,
            data_path_original=data_path_original,
            batch_size=batch_size,
            val_split_ratio=val_split_ratio, # Needed to select the correct split
            is_distributed=is_distributed,
            is_train=False, # Key difference for testing/validation
            num_workers=num_workers,
            seed=seed,
        )
    elif dataset == "galaxy":
        # Keep original galaxy loading logic if needed
        load_data = galaxy.load_data
        return load_data(
             # Pass original galaxy args here...
            data_dir=data_dir, # You would need to add data_dir back if using galaxy
            data_info_list_path=data_info_list_path, # Add back if using galaxy
            batch_size=batch_size,
            random_flip=random_flip, # Add back if using galaxy
            is_distributed=is_distributed,
            is_train=False, # Key difference
            post_process=None,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")