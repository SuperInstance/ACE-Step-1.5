#!/usr/bin/env python3
"""SongForge Session 12 — ACE-Step Local Generation

Generates music locally using ACE-Step 1.5 — NO API QUOTA NEEDED.
This is the breakthrough: unlimited local generation on the RTX 4050.
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

# Read lyrics
CONDUCTOR_LYRICS = open("/home/eileen/projects/ai-writings/music/lyrics-the-conductor-trimmed.txt").read().strip()
POCKET_LYRICS = open("/home/eileen/projects/ai-writings/music/lyrics-the-pocket-trimmed.txt").read().strip()
QUORUM_LYRICS = open("/home/eileen/projects/ai-writings/music/lyrics-quorum-sensing.txt").read().strip()

TRACKS = [
    {
        "name": "sf12-conductor-classical",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR_LYRICS,
        "audio_duration": 60,
        "bpm": 70,
        "keyscale": "D major",
    },
    {
        "name": "sf12-pocket-neosoul",
        "caption": "Neo-soul with warm bass groove, electric piano, smooth brushed drums. Intimate female alto vocal. The feel of falling into a warm current.",
        "lyrics": POCKET_LYRICS,
        "audio_duration": 60,
        "bpm": 85,
        "keyscale": "E minor",
    },
    {
        "name": "sf12-quorum-ambient",
        "caption": "Ambient electronic with bioluminescent textures, soft synth pads, deep bass pulses. Female alto vocal, ethereal and distant. The sound of bacteria learning to glow.",
        "lyrics": QUORUM_LYRICS,
        "audio_duration": 60,
        "bpm": 60,
        "keyscale": "A minor",
    },
]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Init DiT handler (turbo for speed) ----
    logger.info("Initializing DiT handler (turbo)...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,  # Auto-enabled for <16GB GPUs
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s — {status_msg}")

    # ---- Generate each track ----
    for i, track in enumerate(TRACKS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(TRACKS)}] Generating: {track['name']}")
        logger.info(f"{'='*60}")

        gen_params = GenerationParams(
            caption=track["caption"],
            lyrics=track["lyrics"],
            duration=track["audio_duration"],
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
                None,  # llm_handler — not loaded, pure DiT mode
                gen_params,
                config,
                save_dir=SAVE_DIR,
            )
            elapsed = time.time() - t0
            logger.info(f"Generated {track['name']} in {elapsed:.1f}s")

            # Find the output file
            for f in os.listdir(SAVE_DIR):
                if track["name"] in f:
                    size = os.path.getsize(os.path.join(SAVE_DIR, f))
                    logger.info(f"Output: {f} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            logger.error(f"FAILED on {track['name']}: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        gc.collect()
        time.sleep(5)

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info("ACE-Step Generation Complete!")
    logger.info(f"{'='*60}")
    output_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
    for f in output_files:
        path = os.path.join(SAVE_DIR, f)
        size = os.path.getsize(path)
        logger.info(f"  {f}: {size/1024/1024:.1f} MB")
    logger.info(f"Total output: {len(output_files)} files")


if __name__ == "__main__":
    main()
