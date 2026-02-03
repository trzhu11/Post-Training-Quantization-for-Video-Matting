import os
import random
from torch.utils.data import Dataset
from PIL import Image
from PIL import Image
from PIL import ImageFile

from .augmentation import MotionAugmentation

class VideoMatteDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(os.listdir(os.path.join(videomatte_dir, 'fgr')))
        self.videomatte_frames = [sorted(os.listdir(os.path.join(videomatte_dir, 'fgr', clip))) 
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = []
        # target_frames = list(range(seq_length)) # 目标帧索引
        target_frames = [i * 2 for i in range(seq_length)]
        for clip_idx in range(len(self.videomatte_clips)):
            # 检查视频是否足够长
            if len(self.videomatte_frames[clip_idx]) > max(target_frames):
                # 每个clip只生成一个样本，包含全部目标帧
                self.videomatte_idx.append((
                    clip_idx, 
                    target_frames  # 直接存储目标帧列表
                ))
            else:
                print(f"跳过 {self.videomatte_clips[clip_idx]}，帧数不足")
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.videomatte_idx)
    
    def __getitem__(self, idx):
        bgrs = self._get_random_video_background()
        fgrs, phas = self._get_videomatte(idx)
        
        if self.transform is not None:
            return self.transform(fgrs, phas, bgrs)
        
        return fgrs, phas, bgrs
    
    def _get_random_video_background(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载被截断的图片
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_videomatte(self, idx):
        clip_idx, frame_indices = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        fgrs, phas = [], []
        for frame_idx in frame_indices:
            frame = self.videomatte_frames[clip_idx][frame_idx]
            with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )