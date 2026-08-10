#!/usr/bin/env python3
"""SongForge Session 29 — Vocal BPM Study via ACE-Step

Monday 6:46 AM AKST, August 10 2026. MMX daily quota exhausted (status 2).
Weekly quota at 69% but daily gate blocks all MMX calls.
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

PRIMARY EXPERIMENT: Vocal BPM Study
Same lyrics, same key (C major), same prompt caption, 6 different BPMs.
Tests whether the bimodal curve found in instrumental MMX tracks persists:
  - With vocals
  - On a different generation system (ACE-Step vs MMX)

BPM points: 40, 60, 80, 100, 120, 140

SECONDARY: Two new impossible genre fusions as a treat.
"""

import json
import os
import sys
import time
import gc
import traceback

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
LYRICS_DIR = "/home/eileen/projects/ai-writings/music"

def load_lyrics(name):
    path = os.path.join(LYRICS_DIR, name)
    with open(path) as f:
        return f.read().strip()

# Vocal BPM study lyrics
VOCAL_BPM_LYRICS = load_lyrics("lyrics-vocal-bpm-study.txt")

# ═══════════ EXPERIMENT A: Vocal BPM Study ═══════════
# Same as MMX instrumental study but WITH vocals
# 6 BPMs: 40, 60, 80, 100, 120, 140
BPM_CAPTION = "Warm indie folk, fingerpicked acoustic guitar, soft piano, gentle female alto vocals, intimate and conversational"

EXPERIMENT_A_TRACKS = [
    {"name": "sf29-vocal-bpm-040", "bpm": 40,  "duration": 60},
    {"name": "sf29-vocal-bpm-060", "bpm": 60,  "duration": 60},
    {"name": "sf29-vocal-bpm-080", "bpm": 80,  "duration": 60},
    {"name": "sf29-vocal-bpm-100", "bpm": 100, "duration": 60},
    {"name": "sf29-vocal-bpm-120", "bpm": 120, "duration": 60},
    {"name": "sf29-vocal-bpm-140", "bpm": 140, "duration": 60},
]

# ═══════════ EXPERIMENT B: New Impossible Fusions ═══════════
EXPERIMENT_B_TRACKS = [
    {
        "name": "sf29-free-jazz-balkan-brass",
        "caption": "Free jazz balkan brass, chaotic horns, odd meters, Fanfare Ciocarlia meets Ornette Coleman, celebration and deconstruction, raw energy",
        "lyrics": """[Verse]
The trumpet knows a secret it won't tell
The tuba has been drinking since the bell
Rang out in odd-time celebration
The saxophone forgot the notation

[Chorus]
The village burns the brass plays on
The form dissolves the rhythm's gone
The dance doesn't need a key
The free jazz balkan brass agrees""",
        "duration": 60,
        "bpm": 150,
        "keyscale": "F minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf29-ambient-blackgaze-dub",
        "caption": "Ambient blackgaze dub, atmospheric black metal guitars meeting King Tubby dub production, vast reverb spaces, tape delays, tremolo picking dissolving into echoes",
        "lyrics": """[Verse]
The frost forms on the mixing board
The delay pedal catches a chord
That was meant for a cathedral
But ends up in a winter forest
Where every echo is a ghost

[Chorus]
The black metal meets the dub
The blast beat meets the one-drop
The reverb is the same religion
The cold and the bass are one""",
        "duration": 60,
        "bpm": 70,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf29-gamelan-techno",
        "caption": "Microtone gamelan techno, Indonesian pelog and slendro scales meeting Detroit minimal techno, metallic percussion, looping patterns shifting in phase",
        "lyrics": """[Verse]
The bronze keys sing in frequencies
Between the notes we know
The detuned pairs beat slowly
Like waves that come and go

[Chorus]
The palace is a warehouse now
The gongs are drum machines
The pattern shifts a cent per loop
The old scale always redeems""",
        "duration": 60,
        "bpm": 128,
        "keyscale": "C minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

def run_generation(handler, tracks, experiment_name):
    """Generate a batch of tracks."""
    results = []
    for i, track in enumerate(tracks):
        name = track["name"]
        logger.info(f"[{experiment_name}] Track {i+1}/{len(tracks)}: {name}")

        caption = track.get("caption", BPM_CAPTION)
        lyrics = track.get("lyrics", VOCAL_BPM_LYRICS)
        duration = track.get("duration", 60)
        bpm = track.get("bpm", 80)
        keyscale = track.get("keyscale", "C major")
        inference_steps = track.get("inference_steps", 8)
        guidance_scale = track.get("guidance_scale", 7.0)  # Will be overridden to 1.0 by turbo

        params = GenerationParams(
            duration=float(duration),
            keyscale=keyscale,
            bpm=bpm,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            caption=caption,
            lyrics=lyrics,
        )

        config = GenerationConfig(
            batch_size=1,
            audio_format="mp3",
            mp3_bitrate="256k",
        )

        try:
            start_time = time.time()
            result_obj = generate_music(handler, None, params, config, save_dir=SAVE_DIR)
            elapsed = time.time() - start_time

            # Find output file
            full_path = None
            if isinstance(result_obj, dict) and "audio_path" in result_obj:
                full_path = result_obj["audio_path"]
            elif isinstance(result_obj, str):
                full_path = result_obj
            else:
                # Look for most recent file in save_dir
                import glob
                files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.mp3")), key=os.path.getmtime, reverse=True)
                if files:
                    full_path = files[0]
            
            size_mb = os.path.getsize(full_path) / (1024 * 1024) if full_path and os.path.exists(full_path) else 0

            result = {
                "name": name,
                "path": full_path or "unknown",
                "size_mb": round(size_mb, 2),
                "time": round(elapsed, 1),
                "bpm": bpm,
                "key": keyscale,
                "duration": duration,
            }
            results.append(result)
            logger.info(f"  \u2713 {name}: {size_mb:.2f}MB in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")
            results.append({"name": name, "error": str(e)})

        gc.collect()

    return results


def main():
    logger.info("=" * 60)
    logger.info("SongForge Session 29: Vocal BPM Study + New Fusions")
    logger.info("=" * 60)

    # Initialize handler with model loading
    handler = AceStepHandler()
    logger.info("Loading ACE-Step turbo model...")
    t0 = time.time()
    status_msg, success = handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,
    )
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    logger.info(f"Status: {status_msg}")

    if not success:
        logger.error("Model load failed! Cannot proceed with generation.")
        return

    all_results = {}

    # Experiment A: Vocal BPM Study
    logger.info("")
    logger.info("═══════ EXPERIMENT A: Vocal BPM Study ═══════")
    logger.info("Same lyrics, same key (C major), same caption, 6 BPMs")
    logger.info("Testing: does bimodal curve persist with vocals on ACE-Step?")
    logger.info("")

    results_a = run_generation(handler, EXPERIMENT_A_TRACKS, "A")
    all_results["experiment_a_vocal_bpm"] = results_a

    # Print BPM study table
    logger.info("")
    logger.info("VOCAL BPM STUDY RESULTS:")
    logger.info(f"{'BPM':>6} | {'Size (MB)':>10} | {'Time (s)':>8}")
    logger.info("-" * 35)
    for r in results_a:
        if "size_mb" in r:
            logger.info(f"{r['bpm']:>6} | {r['size_mb']:>10.2f} | {r['time']:>8.1f}")
        else:
            logger.info(f"{r['bpm']:>6} | {'ERROR':>10} | {'-':>8}")

    # Experiment B: New Impossible Fusions
    logger.info("")
    logger.info("═══════ EXPERIMENT B: New Impossible Fusions ═══════")
    results_b = run_generation(handler, EXPERIMENT_B_TRACKS, "B")
    all_results["experiment_b_fusions"] = results_b

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION 29 COMPLETE")
    logger.info("=" * 60)
    logger.info(json.dumps(all_results, indent=2))

    # Save results
    results_path = os.path.join(SAVE_DIR, "session29_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
