#!/usr/bin/env python3
"""SongForge Session 14 — ACE-Step Frontier Push

Four experiments extending Session 13's findings:

EXPERIMENT A: 180-Second Duration Test
  Three tracks at 180s — can ACE-Step sustain coherence for 3 minutes?
  If 120s worked cleanly (Session 13), where does it break?

EXPERIMENT B: Explicit Seed Reproducibility
  Same song, SAME EXPLICIT SEED, two runs. Are they identical?
  Session 13 proved no-seed ≠ reproducible. Does explicit seed work?

EXPERIMENT C: Non-Turbo Model (0.6B) — Real Guidance Scale
  Load acestep-5Hz-lm-0.6B (1.3GB) instead of turbo.
  Generate same song at guidance 3.0 and 11.0.
  Does guidance scale ACTUALLY affect output when not overridden?

EXPERIMENT D: New Corpus Adaptations — The Cadence Caller + Buzz of the Yard
  Two essays never before adapted, in genres that test the model's range.

Total: ~10 generations. Zero API calls. Zero quota consumed.
Estimated time: ~25-35 minutes (longer due to 180s tracks + model swap).
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

# Load lyrics
CADENCE = load_lyrics("lyrics-the-cadence-caller-trimmed.txt")
BUZZ = load_lyrics("lyrics-the-buzz-of-the-yard-trimmed.txt")
CONDUCTOR = load_lyrics("lyrics-the-conductor-trimmed.txt")
ENSIGN_V2 = load_lyrics("lyrics-the-ensign-counts-stars-v2-trimmed.txt")
SCHEDULER = load_lyrics("lyrics-the-scheduler-hears-trimmed.txt")

# ═══════════════════════════════════════════════════════
# EXPERIMENT A: 180-Second Duration Push
# ═══════════════════════════════════════════════════════
EXPERIMENT_A = [
    {
        "name": "sf14-duration-180-ambient",
        "caption": "Deep ambient drone, slowly evolving textures, sub-bass frequencies, distant choir. The sound of tectonic plates having a conversation. No rush, no climax, just presence.",
        "lyrics": "",  # instrumental
        "duration": 180,
        "bpm": 50,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf14-duration-180-folk",
        "caption": "Contemporary indie folk, fingerpicked acoustic guitar, soft cello, intimate female alto vocal. Long-form, slowly unfolding story. The patience of a long night watch.",
        "lyrics": ENSIGN_V2,
        "duration": 180,
        "bpm": 65,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf14-duration-180-jazz",
        "caption": "Cool jazz, Miles Davis style, trumpet with mute, upright bass, brushed drums, Rhodes piano. Spacious and patient. The long form of a sunset.",
        "lyrics": CADENCE,
        "duration": 180,
        "bpm": 70,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════════════════════════════════════════════════
# EXPERIMENT B: Explicit Seed Reproducibility
# Same song, explicit seed=42, two runs
# ═══════════════════════════════════════════════════════
SEED_REPRO_PARAMS = {
    "name": "sf14-seed-repro",
    "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
    "lyrics": CONDUCTOR,
    "duration": 60,
    "bpm": 70,
    "keyscale": "D major",
    "inference_steps": 8,
    "guidance_scale": 7.0,
    "seed": 42,
}

# ═══════════════════════════════════════════════════════
# EXPERIMENT C: Non-Turbo Model (0.6B) — Real Guidance Scale
# ═══════════════════════════════════════════════════════
NON_TURBO_TRACKS = [
    {
        "name": "sf14-nonturbo-guidance-03",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 20,  # non-turbo needs more steps
        "guidance_scale": 3.0,
    },
    {
        "name": "sf14-nonturbo-guidance-11",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 20,
        "guidance_scale": 11.0,
    },
]

# ═══════════════════════════════════════════════════════
# EXPERIMENT D: New Corpus Adaptations
# ═══════════════════════════════════════════════════════
EXPERIMENT_D = [
    {
        "name": "sf14-cadence-caller-jazz",
        "caption": "Cool jazz, trumpet with harmon mute, upright bass walking, brushed drums, Rhodes piano. Spacious, patient, Miles Davis meets Erik Satie. Female alto vocal, intimate and conversational.",
        "lyrics": CADENCE,
        "duration": 60,
        "bpm": 72,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf14-buzz-of-the-yard-industrial",
        "caption": "Industrial folk, metallic percussion, anvil strikes, distorted bass, acoustic guitar. The sound of a salvage yard at golden hour. Raw, primal, beautiful. Male baritone vocal, storytelling.",
        "lyrics": BUZZ,
        "duration": 60,
        "bpm": 88,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]


def file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_track(dit_handler, track, suffix=""):
    name = track["name"] + suffix
    logger.info(f"  Generating: {name}")
    logger.info(f"  Duration: {track['duration']}s | BPM: {track['bpm']} | Key: {track['keyscale']} | Guidance: {track['guidance_scale']}")

    gen_params = GenerationParams(
        caption=track["caption"],
        lyrics=track["lyrics"],
        duration=track["duration"],
        bpm=track["bpm"],
        keyscale=track["keyscale"],
        inference_steps=track["inference_steps"],
        guidance_scale=track["guidance_scale"],
    )
    # Set seed if provided
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
        dit_handler,
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
    before_files = set(os.listdir(SAVE_DIR))
    timings = {}

    # ─── EXPERIMENT A: 180s Duration (turbo) ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT A: 180-SECOND DURATION PUSH")
    logger.info(f"{'='*60}")

    logger.info("Loading turbo model for Experiments A, B, D...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,
    )
    if not success:
        logger.error(f"Turbo init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"Turbo loaded in {time.time() - t0:.1f}s")

    timings["A"] = {}
    for track in EXPERIMENT_A:
        try:
            elapsed = generate_track(dit_handler, track)
            timings["A"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # ─── EXPERIMENT B: Seed Reproducibility (turbo) ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT B: EXPLICIT SEED REPRODUCIBILITY")
    logger.info(f"{'='*60}")
    timings["B"] = {}
    for suffix in ["-run1", "-run2"]:
        try:
            elapsed = generate_track(dit_handler, SEED_REPRO_PARAMS, suffix=suffix)
            timings["B"][SEED_REPRO_PARAMS["name"] + suffix] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # Compare hashes
    logger.info("\n--- Seed Reproducibility Check ---")
    run1_files = [f for f in os.listdir(SAVE_DIR) if "sf14-seed-repro-run1" in f and f.endswith(".mp3")]
    run2_files = [f for f in os.listdir(SAVE_DIR) if "sf14-seed-repro-run2" in f and f.endswith(".mp3")]
    if run1_files and run2_files:
        h1 = file_hash(os.path.join(SAVE_DIR, run1_files[0]))
        h2 = file_hash(os.path.join(SAVE_DIR, run2_files[0]))
        logger.info(f"  Run 1 hash: {h1[:32]}...")
        logger.info(f"  Run 2 hash: {h2[:32]}...")
        logger.info(f"  Identical: {'YES' if h1 == h2 else 'NO'}")
        timings["B"]["identical"] = h1 == h2
        timings["B"]["hash1"] = h1[:32]
        timings["B"]["hash2"] = h2[:32]
    else:
        logger.warning("  Could not find both runs")

    # ─── EXPERIMENT D: Corpus Adaptations (turbo) ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT D: NEW CORPUS ADAPTATIONS")
    logger.info(f"{'='*60}")
    timings["D"] = {}
    for track in EXPERIMENT_D:
        try:
            elapsed = generate_track(dit_handler, track)
            timings["D"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # ─── EXPERIMENT C: Non-Turbo Model ───
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT C: NON-TURBO MODEL (0.6B) — REAL GUIDANCE SCALE")
    logger.info(f"{'='*60}")

    # Free turbo model memory
    del dit_handler
    gc.collect()

    logger.info("Loading non-turbo model acestep-5Hz-lm-0.6B (1.3 GB)...")
    t0 = time.time()
    dit_handler_nt = AceStepHandler()
    status_msg, success = dit_handler_nt.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-5Hz-lm-0.6B",
        device="auto",
        offload_to_cpu=True,
    )
    if not success:
        logger.error(f"Non-turbo init failed: {status_msg}")
        logger.warning("Skipping Experiment C — continuing with summary")
    else:
        logger.info(f"Non-turbo (0.6B) loaded in {time.time() - t0:.1f}s")
        timings["C"] = {}
        for track in NON_TURBO_TRACKS:
            try:
                elapsed = generate_track(dit_handler_nt, track)
                timings["C"][track["name"]] = elapsed
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            time.sleep(3)

    # ═══ SUMMARY ═══
    logger.info(f"\n{'='*60}")
    logger.info("SESSION 14 COMPLETE")
    logger.info(f"{'='*60}")

    after_files = set(os.listdir(SAVE_DIR))
    new_files = after_files - before_files
    total_size = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in new_files)

    for f in sorted(new_files):
        size = os.path.getsize(os.path.join(SAVE_DIR, f))
        logger.info(f"  {f}: {size/1024/1024:.1f} MB")

    logger.info(f"\nTotal new files: {len(new_files)}")
    logger.info(f"Total new size: {total_size/1024/1024:.1f} MB")

    timings_path = os.path.join(SAVE_DIR, "session14_timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    logger.info(f"Timings saved to {timings_path}")


if __name__ == "__main__":
    main()
