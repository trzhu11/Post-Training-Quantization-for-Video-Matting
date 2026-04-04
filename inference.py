"""
Inference script for PTQ4VM quantized video matting models.

Usage:
    # Inference on VideoMatte240K test set (low-resolution)
    python inference.py \
        --checkpoint saved_models/quantized_rvm_w4a8.pth \
        --input-root data/videomatte_512x288 \
        --output-root results/w4a8 \
        --device cuda:0

    # Inference on single video
    python inference.py \
        --checkpoint saved_models/quantized_rvm_w4a8.pth \
        --input-source input.mp4 \
        --output-alpha alpha.mp4 \
        --output-foreground foreground.mp4 \
        --output-type video \
        --device cuda:0
"""

import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from quantization.state import enable_quantization


def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):

    assert downsample_ratio is None or (0 < downsample_ratio <= 1)
    assert any([output_composition, output_alpha, output_foreground])
    assert output_type in ['video', 'png_sequence']

    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)

    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(output_composition, frame_rate, int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(output_alpha, frame_rate, int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(output_foreground, frame_rate, int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src_batch in reader:
                src_batch = src_batch.to(device, dtype, non_blocking=True)
                for i in range(src_batch.shape[0]):
                    frame = src_batch[i:i+1]
                    h, w = frame.shape[2:]
                    dr = downsample_ratio if downsample_ratio is not None else auto_downsample_ratio(h, w)
                    fgr, pha, *rec = model(frame, *rec, downsample_ratio=dr)

                    if output_foreground is not None:
                        writer_fgr.write(fgr[0].cpu())
                    if output_alpha is not None:
                        writer_pha.write(pha[0].cpu())
                    if output_composition is not None:
                        if output_type == 'video':
                            com = fgr * pha + bgr * (1 - pha)
                        else:
                            fgr_out = fgr * pha.gt(0)
                            com = torch.cat([fgr_out, pha], dim=-3)
                        writer_com.write(com[0].cpu())

                    bar.update(1)
    finally:
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def load_quantized_model(checkpoint, device):
    model = torch.load(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    enable_quantization(model)
    return model


def inference_dataset(model, input_root, output_root, device, args):
    """Run inference on a dataset with structure: input_root/subset/clip/com/"""
    for subset in sorted(os.listdir(input_root)):
        subset_path = os.path.join(input_root, subset)
        if not os.path.isdir(subset_path):
            continue

        for clip in sorted(os.listdir(subset_path)):
            clip_path = os.path.join(subset_path, clip)
            if not os.path.isdir(clip_path):
                continue

            input_source = os.path.join(clip_path, 'com')
            if not os.path.exists(input_source):
                print(f"Skipping {clip_path}: no 'com' directory found")
                continue

            output_alpha = os.path.join(output_root, subset, clip, 'pha')
            output_foreground = os.path.join(output_root, subset, clip, 'fgr')
            os.makedirs(output_alpha, exist_ok=True)
            os.makedirs(output_foreground, exist_ok=True)

            try:
                convert_video(
                    model,
                    input_source=input_source,
                    input_resize=args.input_resize,
                    downsample_ratio=args.downsample_ratio,
                    output_type='png_sequence',
                    output_alpha=output_alpha,
                    output_foreground=output_foreground,
                    seq_chunk=args.seq_chunk,
                    num_workers=args.num_workers,
                    progress=True,
                    device=device,
                    dtype=torch.float32
                )
                print(f"Done: {subset}/{clip}")
            except Exception as e:
                print(f"Error processing {subset}/{clip}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTQ4VM Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to quantized model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0')

    # Dataset mode
    parser.add_argument('--input-root', type=str, default=None, help='Root directory of input dataset')
    parser.add_argument('--output-root', type=str, default=None, help='Root directory for output results')

    # Single video/sequence mode
    parser.add_argument('--input-source', type=str, default=None, help='Input video file or image sequence directory')
    parser.add_argument('--output-alpha', type=str, default=None)
    parser.add_argument('--output-foreground', type=str, default=None)
    parser.add_argument('--output-composition', type=str, default=None)
    parser.add_argument('--output-type', type=str, default='png_sequence', choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)

    # Common options
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float, default=None)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_quantized_model(args.checkpoint, args.device)

    if args.input_root and args.output_root:
        # Dataset mode
        inference_dataset(model, args.input_root, args.output_root, args.device, args)
    elif args.input_source:
        # Single video/sequence mode
        convert_video(
            model,
            input_source=args.input_source,
            input_resize=args.input_resize,
            downsample_ratio=args.downsample_ratio,
            output_type=args.output_type,
            output_composition=args.output_composition,
            output_alpha=args.output_alpha,
            output_foreground=args.output_foreground,
            output_video_mbps=args.output_video_mbps,
            seq_chunk=args.seq_chunk,
            num_workers=args.num_workers,
            device=args.device,
            dtype=torch.float32
        )
    else:
        print("Error: Provide either --input-root/--output-root or --input-source")
