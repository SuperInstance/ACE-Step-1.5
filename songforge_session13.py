#!/usr/bin/env python3
"""SongForge Session 13 — ACE-Step Deep Experiments

Four experiments that are ONLY possible because of local generation:

EXPERIMENT A: Guidance Scale Sweep
  Same song (The Conductor), 5 guidance scales: 3.0, 5.0, 7.0, 11.0, 15.0
  How does guidance scale affect musical quality and creativity?

EXPERIMENT B: Duration Push
  Three songs at 120s (double the previous max)
  Can ACE-Step sustain coherence over 2 minutes?

EXPERIMENT C: New Corpus Adaptations
  Three new essays adapted: Scheduler Hears, Instanton in Coltrane, Ensign Counts Stars
  Each in a genre chosen to match the essay's character

EXPERIMENT D: Seed Reproducibility
  Same song, same seed, two runs. Is output identical?
  (Run the same params twice, compare file hashes)

Total: ~13 generations × ~90s each = ~20 minutes of GPU time.
Zero API calls. Zero quota consumed.
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

# ─── LOAD LYRICS ───
LYRICS_DIR = "/home/eileen/projects/ai-writings/music"

def load_lyrics(name):
    path = os.path.join(LYRICS_DIR, name)
    with open(path) as f:
        return f.read().strip()

CONDUCTOR = load_lyrics("lyrics-the-conductor-trimmed.txt")
SCHEDULER = load_lyrics("lyrics-the-scheduler-hears-trimmed.txt")
INSTANTON = load_lyrics("lyrics-the-instanton-trimmed.txt")
ENSIGN = load_lyrics("lyrics-the-ensign-counts-stars-trimmed.txt")
POCKET = load_lyrics("lyrics-the-pocket-trimmed.txt")
QUORUM = load_lyrics("lyrics-quorum-sensing.txt")

# ═══════════════════════════════════════════════════════
# EXPERIMENT A: Guidance Scale Sweep
# ═══════════════════════════════════════════════════════
EXPERIMENT_A = [
    {
        "name": "sf13-guidance-03",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 3.0,
    },
    {
        "name": "sf13-guidance-05",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 5.0,
    },
    {
        "name": "sf13-guidance-07",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf13-guidance-11",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 11.0,
    },
    {
        "name": "sf13-guidance-15",
        "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
        "lyrics": CONDUCTOR,
        "duration": 60,
        "bpm": 70,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 15.0,
    },
]

# ═══════════════════════════════════════════════════════
# EXPERIMENT B: Duration Push (120 seconds)
# ═══════════════════════════════════════════════════════
EXPERIMENT_B = [
    {
        "name": "sf13-duration-120-quorum",
        "caption": "Ambient electronic with bioluminescent textures, soft synth pads, deep bass pulses. Female alto vocal, ethereal and distant. The sound of bacteria learning to glow. Long-form, slowly evolving.",
        "lyrics": QUORUM,
        "duration": 120,
        "bpm": 60,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf13-duration-120-scheduler",
        "caption": "Minimalist electronic, rhythmic pulses like a heartbeat, ticking clock textures, subtle synth pads. Male spoken-word vocal. The sound of automation keeping things alive.",
        "lyrics": SCHEDULER,
        "duration": 120,
        "bpm": 120,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf13-duration-120-pocket",
        "caption": "Neo-soul with warm bass groove, electric piano, smooth brushed drums. Intimate female alto vocal. The feel of falling into a warm current. Extended groove.",
        "lyrics": POCKET,
        "duration": 120,
        "bpm": 85,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════════════════════════════════════════════════
# EXPERIMENT C: New Corpus Adaptations
# ═══════════════════════════════════════════════════════
EXPERIMENT_C = [
    {
        "name": "sf13-scheduler-minimalist",
        "caption": "Minimalist post-rock, Steve Reich meets Godspeed You Black Emperor. Pulsing rhythms, layered guitars, building tension through repetition. Male baritone vocal, precise and detached.",
        "lyrics": SCHEDULER,
        "duration": 60,
        "bpm": 120,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf13-instanton-jazz",
        "caption": "Modal jazz in the style of John Coltrane's later work. Soprano saxophone leads, piano comping, elastic time feel, searching and spiritual. Female alto vocal, scat and wordless.",
        "lyrics": INSTANTON,
        "duration": 60,
        "bpm": 140,
        "keyscale": "F minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf13-ensign-folk",
        "caption": "Contemporary indie folk, fingerpicked acoustic guitar, soft cello, intimate male tenor vocal. The feel of standing night watch alone under a sky full of stars.",
        "lyrics": ENSIGN,
        "duration": 60,
        "bpm": 65,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════════════════════════════════════════════════
# EXPERIMENT D: Seed Reproducibility
# Same song, same seed — run twice, compare hashes
# ═══════════════════════════════════════════════════════
REPRO_PARAMS = {
    "name": "sf13-repro",
    "caption": "Classical orchestral, building from whispered strings through woodwind solos to full brass and percussion crescendo. Emotional, cinematic, intimate then overwhelming. Male baritone vocal.",
    "lyrics": CONDUCTOR,
    "duration": 60,
    "bpm": 70,
    "keyscale": "D major",
    "inference_steps": 8,
    "guidance_scale": 7.0,
}

ALL_TRACKS = (
    [("A", t) for t in EXPERIMENT_A] +
    [("B", t) for t in EXPERIMENT_B] +
    [("C", t) for t in EXPERIMENT_C]
)


def generate_track(dit_handler, track, suffix=""):
    """Generate a single track and return timing info."""
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


def file_hash(filepath):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ─── Track existing files ───
    before_files = set(os.listdir(SAVE_DIR))

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

    timings = {}

    # ═══════════════════════════════════════════════════
    # EXPERIMENT A: Guidance Scale Sweep
    # ═══════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT A: GUIDANCE SCALE SWEEP")
    logger.info(f"{'='*60}")
    logger.info("Same song (The Conductor), 5 guidance scales: 3.0, 5.0, 7.0, 11.0, 15.0")
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

    # ═══════════════════════════════════════════════════
    # EXPERIMENT B: Duration Push
    # ═══════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT B: DURATION PUSH (120 seconds)")
    logger.info(f"{'='*60}")
    logger.info("Can ACE-Step sustain coherence over 2 minutes?")
    timings["B"] = {}
    for track in EXPERIMENT_B:
        try:
            elapsed = generate_track(dit_handler, track)
            timings["B"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # ═══════════════════════════════════════════════════
    # EXPERIMENT C: New Corpus Adaptations
    # ═══════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT C: NEW CORPUS ADAPTATIONS")
    logger.info(f"{'='*60}")
    logger.info("Three new essays: Scheduler (minimalist), Instanton (jazz), Ensign (folk)")
    timings["C"] = {}
    for track in EXPERIMENT_C:
        try:
            elapsed = generate_track(dit_handler, track)
            timings["C"][track["name"]] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # ═══════════════════════════════════════════════════
    # EXPERIMENT D: Seed Reproducibility
    # ═══════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT D: SEED REPRODUCIBILITY")
    logger.info(f"{'='*60}")
    logger.info("Same song, run twice. Compare hashes for identical output.")
    timings["D"] = {}
    for suffix in ["-run1", "-run2"]:
        try:
            elapsed = generate_track(dit_handler, REPRO_PARAMS, suffix=suffix)
            timings["D"][REPRO_PARAMS["name"] + suffix] = elapsed
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        time.sleep(3)

    # ─── Compare hashes for reproducibility ───
    logger.info("\n--- Reproducibility Check ---")
    run1_files = [f for f in os.listdir(SAVE_DIR) if "sf13-repro-run1" in f and f.endswith(".mp3")]
    run2_files = [f for f in os.listdir(SAVE_DIR) if "sf13-repro-run2" in f and f.endswith(".mp3")]
    if run1_files and run2_files:
        h1 = file_hash(os.path.join(SAVE_DIR, run1_files[0]))
        h2 = file_hash(os.path.join(SAVE_DIR, run2_files[0]))
        logger.info(f"  Run 1 hash: {h1[:16]}...")
        logger.info(f"  Run 2 hash: {h2[:16]}...")
        logger.info(f"  Identical: {'YES' if h1 == h2 else 'NO'}")
        timings["D"]["identical"] = h1 == h2
        timings["D"]["hash1"] = h1[:16]
        timings["D"]["hash2"] = h2[:16]
    else:
        logger.warning("  Could not find both runs for comparison")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("SESSION 13 COMPLETE")
    logger.info(f"{'='*60}")

    after_files = set(os.listdir(SAVE_DIR))
    new_files = after_files - before_files
    total_size = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in new_files)

    for f in sorted(new_files):
        size = os.path.getsize(os.path.join(SAVE_DIR, f))
        logger.info(f"  {f}: {size/1024/1024:.1f} MB")

    logger.info(f"\nTotal new files: {len(new_files)}")
    logger.info(f"Total new size: {total_size/1024/1024:.1f} MB")

    # Save timings
    timings_path = os.path.join(SAVE_DIR, "session13_timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    logger.info(f"Timings saved to {timings_path}")


if __name__ == "__main__":
    main()
