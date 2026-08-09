#!/usr/bin/env python3
"""SongForge Session 15 — The Cross-Pollination Session

Experiments:
A: 240-Second Duration Push (can ACE-Step hold for 4 minutes?)
B: Seed Variance Sweep (how different are different seeds?)
C: Extreme Genre Mashups (Baroque+DnB, ThroatSinging+Synthwave, Blues+KPop, Gregorian+Techno)
D: Cover Reference Tracks (clean tracks designed for future MMX re-covering)
"""

import json
import os
import sys
import time
import gc
import hashlib

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

CONDUCTOR = load_lyrics("lyrics-the-conductor-trimmed.txt")
SCHEDULER = load_lyrics("lyrics-the-scheduler-hears-trimmed.txt")
POCKET = load_lyrics("lyrics-the-pocket-trimmed.txt")
TENSOR = load_lyrics("lyrics-the-tensor-trimmed.txt")

# ═══════════ EXPERIMENT A: 240s Duration Push ═══════════
EXPERIMENT_A = [
    {
        "name": "sf15-duration-240-ambient",
        "caption": "Deep ambient soundscape, four minutes of slowly evolving texture. Sub-bass drone at 40Hz, layered with shimmering high-frequency partials like sunlight through ice. Occasional distant bell tones. The sound of deep space watching a clock.",
        "lyrics": "",
        "duration": 240,
        "bpm": 40,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf15-duration-240-folk",
        "caption": "Long-form indie folk ballad, four minutes. Fingerpicked acoustic guitar through four verses, soft cello entering in verse 3, harmonium in verse 4. The patience of watching a long sunset from a ship's deck.",
        "lyrics": TENSOR,
        "duration": 240,
        "bpm": 60,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: Seed Variance Sweep ═══════════
SEED_VALUES = [42, 137, 256, 777]
EXPERIMENT_B_BASE = {
    "caption": "Warm indie folk, fingerpicked guitar, soft female alto vocal, gentle cello. A quiet prayer sung in a small room.",
    "lyrics": POCKET,
    "duration": 60,
    "bpm": 75,
    "keyscale": "G major",
    "inference_steps": 8,
    "guidance_scale": 7.0,
}

# ═══════════ EXPERIMENT C: Extreme Genre Mashups ═══════════
EXPERIMENT_C = [
    {
        "name": "sf15-mashup-baroque-dnb",
        "caption": "Baroque chamber music meets drum and bass. Harpsichord and viola da gamba playing a Bach-style fugue, but the rhythm section is 170 BPM amen-break-style drums with sub-bass. The collision of 1720 and 2020. Contrapuntal bass drops.",
        "lyrics": "",
        "duration": 60,
        "bpm": 170,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf15-mashup-throat-singing-synthwave",
        "caption": "Mongolian throat singing (khoomei) over 80s synthwave production. Analog pad synths, gated reverb drums, rolling bassline. The deep overtone singing rides above the neon like a monk in an arcade. Tuvan vocals meets drive-home nostalgia.",
        "lyrics": "",
        "duration": 60,
        "bpm": 110,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf15-mashup-blues-kpop",
        "caption": "Mississippi delta blues guitar (Robert Johnson style, open A tuning) meets K-pop production. Crackle of old 78rpm records, fingerpicked acoustic blues, but the arrangement has modern pop hooks, synth stabs, and a bright chorus. The crossroad where Seoul meets Clarksdale.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 120,
        "keyscale": "A major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf15-mashup-gregorian-techno",
        "caption": "Gregorian chant meets Berlin techno. Monks singing in Latin over a 128 BPM four-on-the-floor kick drum, acid 303 bassline, and dub delays. The cathedral has a strobe light. The prayer is the dancefloor.",
        "lyrics": SCHEDULER,
        "duration": 60,
        "bpm": 128,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: Cover Reference Tracks ═══════════
EXPERIMENT_D = [
    {
        "name": "sf15-cover-ref-clean-folk",
        "caption": "Clean, simple indie folk production. Single acoustic guitar, unadorned female vocal, no effects. Designed as a reference track for genre-morphing covers.",
        "lyrics": POCKET,
        "duration": 60,
        "bpm": 80,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf15-cover-ref-clean-ambient",
        "caption": "Clean ambient pad, slowly modulating, no percussion. Designed as a reference for rhythm-addition covers. Pure texture.",
        "lyrics": "",
        "duration": 60,
        "bpm": 60,
        "keyscale": "F major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]


def generate_track(handler, track, suffix=""):
    name = track["name"] + suffix
    logger.info(f"  Generating: {name}")
    logger.info(f"  Duration: {track['duration']}s | BPM: {track['bpm']} | Key: {track['keyscale']}")

    gen_params = GenerationParams(
        caption=track["caption"],
        lyrics=track["lyrics"],
        duration=track["duration"],
        bpm=track["bpm"],
        keyscale=track["keyscale"],
        inference_steps=track["inference_steps"],
        guidance_scale=track["guidance_scale"],
    )
    if "seed" in track:
        gen_params.seed = track["seed"]
        logger.info(f"  Seed: {track['seed']}")

    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
        mp3_bitrate="256k",
    )

    t0 = time.time()
    result = generate_music(
        handler,
        None,
        gen_params,
        config,
        save_dir=SAVE_DIR,
    )
    elapsed = time.time() - t0
    logger.info(f"  Generated in {elapsed:.1f}s")
    return elapsed


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timings = {"A": {}, "B": {}, "C": {}, "D": {}}

    logger.info(f"\n{'='*60}")
    logger.info("Session 15: The Cross-Pollination Session")
    logger.info(f"{'='*60}")

    # Load model
    logger.info("Loading ACE-Step turbo model...")
    t0 = time.time()
    handler = AceStepHandler()
    status_msg, success = handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,
    )
    if not success:
        logger.error(f"Turbo init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # ─── EXPERIMENT A: 240s Duration ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT A: 240-Second Duration Push")
    logger.info(f"{'='*60}")
    for track in EXPERIMENT_A:
        try:
            elapsed = generate_track(handler, track)
            timings["A"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        if hasattr(handler, '__dict__'):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        time.sleep(3)

    # ─── EXPERIMENT B: Seed Variance ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT B: Seed Variance Sweep")
    logger.info(f"{'='*60}")
    for seed in SEED_VALUES:
        track = {**EXPERIMENT_B_BASE, "name": "sf15-seed-variance", "seed": seed}
        try:
            elapsed = generate_track(handler, track, suffix=f"-seed{seed}")
            timings["B"][f"sf15-seed-variance-seed{seed}"] = elapsed
        except Exception as e:
            logger.error(f"  FAILED seed {seed}: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(3)

    # ─── EXPERIMENT C: Genre Mashups ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT C: Extreme Genre Mashups")
    logger.info(f"{'='*60}")
    for track in EXPERIMENT_C:
        try:
            elapsed = generate_track(handler, track)
            timings["C"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(3)

    # ─── EXPERIMENT D: Cover References ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT D: Cover Reference Tracks")
    logger.info(f"{'='*60}")
    for track in EXPERIMENT_D:
        try:
            elapsed = generate_track(handler, track)
            timings["D"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(3)

    # Save timings
    timings_path = os.path.join(SAVE_DIR, "session15_timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    logger.info(f"\nTimings saved to {timings_path}")

    # Summary
    total_tracks = sum(1 for exp in timings.values() for v in exp.values() if isinstance(v, (int, float)))
    total_errors = sum(1 for exp in timings.values() for v in exp.values() if not isinstance(v, (int, float)))
    total_time = sum(v for exp in timings.values() for v in exp.values() if isinstance(v, (int, float)))
    logger.info(f"\n{'='*60}")
    logger.info(f"Session 15 Complete: {total_tracks} tracks, {total_errors} errors, {total_time:.0f}s total")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
