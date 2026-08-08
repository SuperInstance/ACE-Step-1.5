#!/usr/bin/env python3
"""SongForge Session 12C — Quorum Sensing Multi-Genre Suite

The bacterial communication essay across 5 genres.
Plus a temperature/seed experiment on the conductor.
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

QUORUM_LYRICS = open("/home/eileen/projects/ai-writings/music/lyrics-quorum-sensing.txt").read().strip()

TRACKS = [
    {
        "name": "sf12-quorum-folk",
        "caption": "Indie folk, fingerpicked acoustic guitar, warm female alto vocal. Intimate, like a conversation in the dark. The sound of cells learning to cooperate.",
        "bpm": 72,
        "keyscale": "A minor",
    },
    {
        "name": "sf12-quorum-jazz",
        "caption": "Cool jazz, upright bass walking, brushed snare, muted trumpet. Female alto vocal, smoky and warm. The sound of a colony finding its rhythm.",
        "bpm": 90,
        "keyscale": "D minor",
    },
    {
        "name": "sf12-quorum-choral",
        "caption": "Sacred choral music, unaccompanied voices in a cathedral. Female soprano lead with choir. The sound of thousands of cells singing in unison.",
        "bpm": 60,
        "keyscale": "C major",
    },
    {
        "name": "sf12-quorum-techno",
        "caption": "Minimal techno, steady four-on-the-floor, analog synth bass, gradual build. The sound of a population density crossing a threshold.",
        "bpm": 128,
        "keyscale": "A minor",
    },
    {
        "name": "sf12-quorum-country",
        "caption": "Alternative country, pedal steel guitar, gentle acoustic strumming. Male baritone vocal, weathered and warm. The bacteria as a metaphor for community.",
        "bpm": 80,
        "keyscale": "G major",
    },
]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

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

    for i, track in enumerate(TRACKS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(TRACKS)}] Generating: {track['name']}")
        logger.info(f"{'='*60}")

        gen_params = GenerationParams(
            caption=track["caption"],
            lyrics=QUORUM_LYRICS,
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
        time.sleep(3)

    logger.info(f"\nQuorum Sensing Suite Complete!")
    output_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.mp3')])
    total_size = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in output_files)
    logger.info(f"Total: {len(output_files)} files, {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
