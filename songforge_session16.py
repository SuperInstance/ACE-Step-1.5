#!/usr/bin/env python3
"""SongForge Session 16 — The Saturday Evening Deep Structure

New experiments with ACE-Step 1.5 turbo on RTX 4050:
A: Guidance Scale Sweep (find the sweet spot — 3.0 to 15.0)
B: New corpus adaptations (The Salvage Choir, Free Energy, Mycorrhizal Network)
C: Extreme Impossible Genres (Klezmer Drum & Bass, Tuvan Throat Singing Shoegaze, Noh Theater Trap)
D: Duration Frontier — 300 seconds (five full minutes)
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
LYRICS_DIR = "/home/eileen/projects/ai-writings/music"

def load_lyrics(name):
    path = os.path.join(LYRICS_DIR, name)
    with open(path) as f:
        return f.read().strip()

SALVAGE = load_lyrics("lyrics-the-salvage-choir.txt")
FREE_ENERGY = load_lyrics("lyrics-the-free-energy-principle.txt")
MYCORRHIZA = load_lyrics("lyrics-the-myocorrhizal-network.txt")

# Use shorter existing lyrics for some experiments
POCKET = load_lyrics("lyrics-the-pocket-trimmed.txt")

# ═══════════ EXPERIMENT A: Guidance Scale Sweep ═══════════
# Systematic sweep from low to high guidance
# Using a consistent prompt to isolate the guidance variable
BASE_CAPTION = "Warm indie folk, fingerpicked acoustic guitar, gentle female alto vocal, soft cello, intimate room sound"
BASE_LYRICS = POCKET
GUIDANCE_VALUES = [3.0, 5.0, 7.0, 9.0, 12.0, 15.0]

EXPERIMENT_A = [
    {
        "name": f"sf16-guidance-{int(g*10):03d}",
        "caption": BASE_CAPTION,
        "lyrics": BASE_LYRICS,
        "duration": 60,
        "bpm": 75,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": g,
    }
    for g in GUIDANCE_VALUES
]

# ═══════════ EXPERIMENT B: New Corpus Adaptations ═══════════
EXPERIMENT_B = [
    {
        "name": "sf16-salvage-choir-industrial-folk",
        "caption": "Industrial folk, metallic percussion, anvil strikes, acoustic guitar, raspy male vocal, sounds of a salvage yard layered into rhythm. The music of work and rust and memory.",
        "lyrics": SALVAGE,
        "duration": 90,
        "bpm": 85,
        "keyscale": "D minor",
        "inference_steps": 10,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf16-free-energy-art-pop",
        "caption": "Art pop, syncopated piano, angular guitar, mathematical structure, jittery drums, then resolving into lush chorus. Like St Vincent producing a neuroscience lecture.",
        "lyrics": FREE_ENERGY,
        "duration": 90,
        "bpm": 110,
        "keyscale": "E minor",
        "inference_steps": 10,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf16-mycorrhiza-ambient-folk",
        "caption": "Deep ambient folk, subterranean bass drones, fingerpicked guitar like roots growing, whispered female vocal, field recordings of soil. The sound of the underground internet.",
        "lyrics": MYCORRHIZA,
        "duration": 90,
        "bpm": 55,
        "keyscale": "C major",
        "inference_steps": 10,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT C: Extreme Impossible Genres ═══════════
EXPERIMENT_C = [
    {
        "name": "sf16-klezmer-dnb",
        "caption": "Klezmer drum and bass. Clarinet wailing in freygish mode over 170 BPM breakbeats, upright bass doubling the synth sub, hora rhythm chopped into amen break. The wedding reception has strobe lights and a mosh pit.",
        "lyrics": "",
        "duration": 60,
        "bpm": 170,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf16-throat-shoegaze",
        "caption": "Tuvan throat singing meets shoegaze. Kargyraa drone vocals layered over My Bloody Valentine walls of guitar, overtone melodies shimmering through distortion. The steppe dissolves into fuzz.",
        "lyrics": "",
        "duration": 60,
        "bpm": 70,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf16-noh-trap",
        "caption": "Noh theater trap music. Nohkan flute piercing over 808 bass and hi-hat triplets, male choral chanting in the background, taiko drums blending with trap snares. The ancient stage meets the cypher.",
        "lyrics": "",
        "duration": 60,
        "bpm": 140,
        "keyscale": "F# minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: Duration Frontier — 300 seconds ═══════════
EXPERIMENT_D = [
    {
        "name": "sf16-duration-300-ambient",
        "caption": "Five-minute deep ambient drift. Sub-bass at 30Hz, slowly evolving harmonics, occasional metallic shimmer like distant ships' bells. The sound of the ocean floor counting time in centuries.",
        "lyrics": "",
        "duration": 300,
        "bpm": 40,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]


def generate_track(handler, track, suffix=""):
    name = track["name"] + suffix
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Generating: {name}")
    logger.info(f"  Duration: {track['duration']}s | BPM: {track['bpm']} | Key: {track['keyscale']}")
    logger.info(f"  Steps: {track['inference_steps']} | Guidance: {track['guidance_scale']}")
    logger.info(f"  Caption: {track['caption'][:100]}...")

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
    try:
        result = generate_music(
            handler,
            None,
            gen_params,
            config,
            save_dir=SAVE_DIR,
        )
        elapsed = time.time() - t0
        logger.info(f"  ✅ Generated in {elapsed:.1f}s")

        # Check output file
        if isinstance(result, dict) and "audio_path" in result:
            fsize = os.path.getsize(result["audio_path"]) / 1024 / 1024
            logger.info(f"  📁 File: {result['audio_path']} ({fsize:.1f}MB)")
        elif isinstance(result, list):
            for r in result:
                if isinstance(r, str) and os.path.exists(r):
                    fsize = os.path.getsize(r) / 1024 / 1024
                    logger.info(f"  📁 File: {r} ({fsize:.1f}MB)")

        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ❌ Failed after {elapsed:.1f}s: {e}")
        return -1


def run_experiment_group(handler, label, title, tracks):
    logger.info(f"\n{'═'*60}")
    logger.info(f"EXPERIMENT {label}: {title}")
    logger.info(f"{'═'*60}")

    timings = {}
    for i, track in enumerate(tracks):
        elapsed = generate_track(handler, track)
        timings[track["name"]] = {
            "elapsed": elapsed,
            "duration": track["duration"],
            "guidance": track["guidance_scale"],
            "bpm": track["bpm"],
        }
        # Cleanup between tracks
        gc.collect()
        if torch_available():
            torch_cache_cleanup()
        time.sleep(2)

    return timings


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def torch_cache_cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_timings = {}

    logger.info(f"\n{'═'*60}")
    logger.info("Session 16: The Saturday Evening Deep Structure")
    logger.info(f"{'═'*60}")

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
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    logger.info(f"Status: {status_msg}")

    if not success:
        logger.error("Model load failed!")
        return

    # ─── Run experiments ───

    # A: Guidance sweep
    all_timings["A_guidance"] = run_experiment_group(
        handler, "A", "Guidance Scale Sweep (3.0 → 15.0)", EXPERIMENT_A
    )

    # B: Corpus adaptations
    all_timings["B_corpus"] = run_experiment_group(
        handler, "B", "New Corpus Adaptations", EXPERIMENT_B
    )

    # C: Impossible genres
    all_timings["C_genres"] = run_experiment_group(
        handler, "C", "Extreme Impossible Genres", EXPERIMENT_C
    )

    # D: Duration frontier
    all_timings["D_duration"] = run_experiment_group(
        handler, "D", "Duration Frontier — 300 Seconds", EXPERIMENT_D
    )

    # Summary
    logger.info(f"\n{'═'*60}")
    logger.info("SESSION 16 SUMMARY")
    logger.info(f"{'═'*60}")
    total_tracks = 0
    total_time = 0
    for group, timings in all_timings.items():
        for name, data in timings.items():
            total_tracks += 1
            if data["elapsed"] > 0:
                total_time += data["elapsed"]
                logger.info(f"  {name}: {data['elapsed']:.1f}s gen ({data['duration']}s audio, guidance {data['guidance']})")
            else:
                logger.info(f"  {name}: FAILED")
    logger.info(f"\nTotal: {total_tracks} tracks, {total_time:.0f}s generation time")

    # Save timings
    timings_path = os.path.join(SAVE_DIR, "sf16-timings.json")
    with open(timings_path, "w") as f:
        json.dump(all_timings, f, indent=2)
    logger.info(f"Timings saved to {timings_path}")


if __name__ == "__main__":
    main()
