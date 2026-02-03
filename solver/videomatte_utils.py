import os
import logging
import time
import yaml
from easydict import EasyDict
import random
import torch
import torchvision.transforms as transforms
from qdrop.solver.videomatte import VideoMatteDataset
from qdrop.solver.videomatte import VideoMatteTrainAugmentation, VideoMatteValidAugmentation
from torch.utils.data import DataLoader
from qdrop.solver.augmentation import TrainFrameSampler, ValidFrameSampler
import numpy as np
logger = logging.getLogger('qdrop')


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            if isinstance(output_batch, tuple):
                self.output_store = output_batch[0]  # 取元组的第一个元素
            else:
                self.output_store = output_batch     # 直接保存单个张量
        if self.stop_forward:
            raise StopForwardException


# load data
def load_data(
    videomatte_dir_train: str,
    background_video_dir_train: str,
    size: int,
    seq_length: int,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
):
    size_lr = (size, size)
    train_transform = VideoMatteTrainAugmentation(size_lr)
    train_sampler = TrainFrameSampler()

    train_dataset = VideoMatteDataset(
        videomatte_dir=videomatte_dir_train,
        background_video_dir=background_video_dir_train,
        size=size,
        seq_length=seq_length,
        seq_sampler=train_sampler,
        transform=train_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info('已完成 VideoMatte 数据集加载')
    return train_loader




