import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs.dstnetplus_deblur_arch import DSTNetPlus_Final


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract cached frames from a video at an offset and run DSTNetPlus deblurring.'
    )
    parser.add_argument('--video', type=str, required=True, help='Input video path.')
    parser.add_argument(
        '--offset',
        type=str,
        required=True,
        help='Timestamp offset for extraction (e.g. "12.5" or "00:00:12.500").',
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=2,
        help='Number of frames to extract starting at --offset. 1 for single-frame deblur.',
    )
    parser.add_argument(
        '--cache_root',
        type=str,
        default='cache/video_frames',
        help='Root folder for cached extracted frames.',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='experiments/DSTNetPlus_pretrained_model/DSTNetPlus_base_gopro.pth',
        help='Path to pretrained checkpoint.',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='results/video_deblur',
        help='Output folder for deblurred images.',
    )
    parser.add_argument(
        '--save_all',
        action='store_true',
        help='Save all deblurred extracted frames. Default saves only the center frame.',
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=30,
        help='Inference chunk size for long frame sequences.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference ("cuda" or "cpu").',
    )
    return parser.parse_args()


def build_cache_key(video_path: Path, offset: str, num_frames: int) -> str:
    stat = video_path.stat()
    payload = {
        'video': str(video_path.resolve()),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'offset': offset,
        'num_frames': num_frames,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def frame_paths(frame_dir: Path, num_frames: int):
    return [frame_dir / f'frame_{i:06d}.png' for i in range(num_frames)]


def extract_frames_cached(video_path: Path, offset: str, num_frames: int, cache_root: Path) -> Path:
    cache_key = build_cache_key(video_path, offset, num_frames)
    cache_dir = cache_root / cache_key
    frames_dir = cache_dir / 'frames'
    meta_path = cache_dir / 'meta.json'
    expected = frame_paths(frames_dir, num_frames)

    if frames_dir.exists() and meta_path.exists() and all(p.exists() for p in expected):
        print(f'Using cached frames: {frames_dir}')
        return frames_dir

    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob('*.png'):
        p.unlink()

    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-ss',
        str(offset),
        '-i',
        str(video_path),
        '-frames:v',
        str(num_frames),
        '-vsync',
        '0',
        str(frames_dir / 'frame_%06d.png'),
    ]
    subprocess.run(cmd, check=True)

    produced = sorted(frames_dir.glob('frame_*.png'))
    if len(produced) == 0:
        raise RuntimeError('No frames were extracted. Check --offset and input video.')

    if len(produced) != num_frames:
        raise RuntimeError(
            f'Extracted {len(produced)} frame(s), expected {num_frames}. '
            'Try reducing --num_frames or using an earlier --offset.'
        )

    meta = {
        'video': str(video_path.resolve()),
        'offset': offset,
        'num_frames': num_frames,
        'cache_key': cache_key,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Extracted and cached frames: {frames_dir}')
    return frames_dir


def load_frames_tensor(frames_dir: Path) -> torch.Tensor:
    paths = sorted(frames_dir.glob('frame_*.png'))
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Failed to read frame: {p}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
    arr = np.stack(imgs, axis=0)  # T,C,H,W
    return torch.from_numpy(arr).unsqueeze(0)  # 1,T,C,H,W


def tensor_to_bgr_uint8(frame_tensor: torch.Tensor) -> np.ndarray:
    arr = frame_tensor.detach().float().cpu().clamp_(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))  # HWC RGB
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = DSTNetPlus_Final(num_feat=64, num_kernel_block=3, num_block=15, nonblind_denoise=False)
    pretrained = torch.load(str(ckpt_path), map_location='cpu')
    state = pretrained['params'] if isinstance(pretrained, dict) and 'params' in pretrained else pretrained
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def infer(model: torch.nn.Module, lq: torch.Tensor, device: torch.device, max_seq_len: int) -> torch.Tensor:
    lq = lq.to(device)
    chunks = []
    with torch.no_grad():
        for i in range(0, lq.size(1), max_seq_len):
            chunks.append(model(lq[:, i:i + max_seq_len]).cpu())
    return torch.cat(chunks, dim=1)


def main():
    args = parse_args()
    video_path = Path(args.video)
    ckpt_path = Path(args.ckpt_path)
    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f'Video not found: {video_path}')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    if args.num_frames <= 0:
        raise ValueError('--num_frames must be > 0')

    requested_device = args.device.lower()
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available; falling back to CPU.')
        requested_device = 'cpu'
    device = torch.device(requested_device)

    frames_dir = extract_frames_cached(video_path, args.offset, args.num_frames, cache_root)
    lq = load_frames_tensor(frames_dir)

    model = load_model(ckpt_path, device)
    try:
        output = infer(model, lq, device, args.max_seq_len)
    except torch.OutOfMemoryError:
        if device.type != 'cuda':
            raise
        print('CUDA OOM during inference. Falling back to CPU inference.')
        torch.cuda.empty_cache()
        cpu_device = torch.device('cpu')
        model = model.to(cpu_device)
        output = infer(model, lq, cpu_device, args.max_seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    offset_tag = args.offset.replace(':', '_').replace('.', '_')

    if args.save_all:
        seq_dir = out_dir / f'{stem}_off_{offset_tag}'
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i in range(output.shape[1]):
            deblur_img = tensor_to_bgr_uint8(output[0, i, ...])
            input_img = tensor_to_bgr_uint8(lq[0, i, ...])
            cv2.imwrite(str(seq_dir / f'deblur_{i:06d}.png'), deblur_img)
            cv2.imwrite(str(seq_dir / f'input_{i:06d}.png'), input_img)
        print(f'Saved {output.shape[1]} deblurred/input frame pair(s) to {seq_dir}')
    else:
        center_idx = output.shape[1] // 2
        deblur_img = tensor_to_bgr_uint8(output[0, center_idx, ...])
        input_img = tensor_to_bgr_uint8(lq[0, center_idx, ...])
        deblur_out_path = out_dir / f'{stem}_off_{offset_tag}_deblur_center.png'
        input_out_path = out_dir / f'{stem}_off_{offset_tag}_input_center.png'
        cv2.imwrite(str(deblur_out_path), deblur_img)
        cv2.imwrite(str(input_out_path), input_img)
        print(f'Saved center deblurred frame to {deblur_out_path}')
        print(f'Saved center input frame to {input_out_path}')


if __name__ == '__main__':
    main()
