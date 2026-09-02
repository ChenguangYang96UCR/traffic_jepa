#!/usr/bin/env python
"""Precompute one frozen CLIP text embedding per window and variable."""
import argparse
import os
import sys

import numpy as np
import torch
from numpy.lib.format import open_memmap
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader import Dataset_Custom
from utils.text_embeddings import clip_cache_path


def build_prompt(name, values):
    values = np.asarray(values, dtype=np.float64)
    slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
    scale = max(float(np.std(values)), 1e-8)
    relative_slope = slope / scale
    if relative_slope > 0.02:
        direction = 'rising'
    elif relative_slope < -0.02:
        direction = 'falling'
    else:
        direction = 'stable'
    return (
        f'Time series variable {name}. Length {len(values)}. '
        f'Mean {np.mean(values):.5g}; standard deviation {np.std(values):.5g}; '
        f'minimum {np.min(values):.5g}; maximum {np.max(values):.5g}; '
        f'first {values[0]:.5g}; last {values[-1]:.5g}; '
        f'trend {direction}; slope {slope:.5g}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--features', default='M', choices=['M', 'S', 'MS'])
    parser.add_argument('--freq', default='h')
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--label_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--model_name', default='openai/clip-vit-base-patch32')
    parser.add_argument('--prompt_batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(args.model_name)
    text_model.eval().requires_grad_(False).to(args.device)
    future_len = max(args.seq_len, args.pred_len)

    for flag in ('train', 'val', 'test'):
        dataset = Dataset_Custom(
            root_path=args.root_path, data_path=args.data_path, flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len, future_len],
            features=args.features, target=args.target, timeenc=1,
            freq=args.freq, text_embedding_dir='')
        output_path = clip_cache_path(
            args.cache_dir, args.data_path, flag, args.seq_len,
            args.pred_len, future_len)
        output = open_memmap(
            output_path, mode='w+', dtype=np.float16,
            shape=(len(dataset), len(dataset.feature_names), 512))

        pending_prompts, pending_locations = [], []

        def flush():
            if not pending_prompts:
                return
            tokens = tokenizer(
                pending_prompts, padding=True, truncation=True,
                max_length=77, return_tensors='pt').to(args.device)
            with torch.no_grad():
                embeddings = text_model(**tokens).text_embeds
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu().numpy().astype(np.float16)
            for embedding, (window_index, variable_index) in zip(
                    embeddings, pending_locations):
                output[window_index, variable_index] = embedding
            pending_prompts.clear()
            pending_locations.clear()

        for window_index in range(len(dataset)):
            start = window_index + args.seq_len
            target_window = dataset.raw_y[start:start + args.seq_len]
            for variable_index, name in enumerate(dataset.feature_names):
                pending_prompts.append(build_prompt(
                    name, target_window[:, variable_index]))
                pending_locations.append((window_index, variable_index))
                if len(pending_prompts) >= args.prompt_batch_size:
                    flush()
            if (window_index + 1) % 250 == 0:
                print(f'{flag}: {window_index + 1}/{len(dataset)} windows')
        flush()
        output.flush()
        print(f'Saved {output.shape} float16 embeddings to {output_path}')


if __name__ == '__main__':
    main()
