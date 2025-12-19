from torch.utils.tensorboard import SummaryWriter

def get_writer(log_dir, experiment_name):
    return SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")