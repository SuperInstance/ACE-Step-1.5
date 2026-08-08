#!/usr/bin/env python3
"""SongForge Session 12B — ACE-Step Genre Matrix Experiment

Tests how the same lyrics sound across 6 different genres.
All local, all quota-free. The constraint is just GPU time.

This is the experiment we could NEVER do with MMX (would cost 6 quota slots).
"""

import json
import os
import sys
import time
import gc

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAVE_DIR = "/home/eileen/projects/ai-writings/music/ace-step-output"

# Use the conductor lyrics — most adaptable
CONDUCTOR_LYRICS = open("/home/eileen/projects/ai-writings/music/lyrics-the-conductor-trimmed.txt").read().strip()

GENRE_MATRIX = [
    {
        "name": "sf12-conductor-classical-v2",
        "caption": "Late Romantic classical, full symphony orchestra. Building from whispered strings to full brass crescendo. Emotional, cinematic.",
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-conductor-deltablues",
        "caption": "Delta blues, acoustic guitar with slide, foot stomp percussion. Raw, bone-dry production. Male vocal with gravel and tenderness.",
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-conductor-dub",
        "caption": "Dub reggae, heavy bass, spring reverb echoes, slow groove, spacious. Drum and bass with tape delay.",
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-conductor-shoegaze",
        "caption": "Shoegaze, wall of guitars, reversed reverbs, buried vocals, dreamlike. My Bloody Valentine meets classical composition.",
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-conductor-acapella",
        "caption": "A cappella, solo male baritone voice in a cathedral. Natural reverb. No instruments at all — just the voice conducting itself.",
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-conductor-synthwave",
        "caption": "Synthwave, retro analog synths, drum machine, neon atmosphere. The conductor reimagined in a neon city at midnight.",
        "bpm": 70,
        "keyscale": "D major",
    },
]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Init DiT handler ----
    logger.info("Initializing DiT handler (turbo)...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s")

    # ---- Generate each track ----
    for i, track in enumerate(GENRE_MATRIX):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(GENRE_MATRIX)}] Generating: {track['name']}")
        logger.info(f"{'='*60}")

        gen_params = GenerationParams(
            caption=track["caption"],
            lyrics=CONDUCTOR_LYRICS,
            duration=60,
            bpm=track["bpm"],
            keyscale=track["keyscale"],
            inference_steps=8,
            guidance_scale=7.0,
        )

        config = GenerationConfig(
            batch_size=1,
            audio_format="mp3",
            mp3_bitrate="256k",
        )

        t0 = time.time()
        try:
            result = generate_music(
                dit_handler,
                None,
                gen_params,
                config,
                save_dir=SAVE_DIR,
            )
            elapsed = time.time() - t0
            logger.info(f"Generated {track['name']} in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"FAILED on {track['name']}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        time.sleep(5)

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info("Genre Matrix Complete!")
    logger.info(f"{'='*60}")
    output_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp3')]
    total_size = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in output_files)
    for f in sorted(output_files):
        path = os.path.join(SAVE_DIR, f)
        size = os.path.getsize(path)
        logger.info(f"  {f}: {size/1024/1024:.1f} MB")
    logger.info(f"Total: {len(output_files)} files, {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
