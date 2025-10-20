# In utils/train_utils/vicddpm_train_util.py
from utils.train_utils.ddpm_train_util import * # Keep this

class VICDDPMTrainLoop(DDPMTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        # batch is expected dict: {'target': tensor, 'condition': tensor}
        target = batch['target']
        condition = batch['condition']
        # Return target tensor and condition dictionary for model_kwargs
        return target, {'condition_input': condition}
