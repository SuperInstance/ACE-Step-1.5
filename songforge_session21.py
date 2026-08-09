#!/usr/bin/env python3
"""SongForge Session 21 — "The Sunday Afternoon Laboratory"

Sunday 1:30 PM AKST. MMX quota still exhausted.
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload). 17GB RAM available.

Six experiments:
A: Feedback Loop Vol. 4 — Session 20 fictions become songs (4 tracks, 90s vocal)
B: Impossible Genre Matrix Vol. 6 — new extreme fusions (4 tracks, 60s instrumental)
C: Prompt Detail Study Vol. 2 — medium prompts across genres (3 tracks, 60s instrumental)
D: Seed Reproducibility Study — same prompt + different seeds (3 tracks, 60s instrumental)
E: Temperature/Guidance Sweep — varying guidance_scale (4 tracks, 60s instrumental)
F: Key Signature Study — same song in different keys (4 tracks, 60s instrumental)
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

# Load lyrics for feedback loop vol.4
POLKA = load_lyrics("lyrics-the-polka-blast-beat.txt")
ZYDECO = load_lyrics("lyrics-the-zydeco-fuzz.txt")
FLAMENCO = load_lyrics("lyrics-the-flamenco-breakbeat.txt")
MARIACHI = load_lyrics("lyrics-the-mariachi-neon.txt")

# ═══════════ EXPERIMENT A: Feedback Loop Vol. 4 ═══════════
EXPERIMENT_A = [
    {
        "name": "sf21-polka-blast-beat",
        "caption": "Polka meets Norwegian black metal, accordion and tremolo guitar, blast beats with oompah bass, folk horror wedding in a frozen church",
        "lyrics": POLKA,
        "duration": 90,
        "bpm": 140,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-zydeco-fuzz",
        "caption": "Zydeco meets shoegaze, washboard and distorted guitar, Cajun accordion through reverb pedals, bayou dream pop",
        "lyrics": ZYDECO,
        "duration": 90,
        "bpm": 82,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-flamenco-breakbeat",
        "caption": "Flamenco meets drum and bass, Spanish guitar and palmas over amen break, compas at 170 BPM, Brixton basement Sevilla",
        "lyrics": FLAMENCO,
        "duration": 90,
        "bpm": 170,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-mariachi-neon",
        "caption": "Mariachi meets synthwave, trumpets through vocoders, guitarron through filter sweeps, Osaka serenata under neon lights",
        "lyrics": MARIACHI,
        "duration": 90,
        "bpm": 100,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: Impossible Genre Matrix Vol. 6 ═══════════
EXPERIMENT_B = [
    {
        "name": "sf21-gagaku-dubstep",
        "caption": "Gagaku court music meets dubstep, sho mouth organ and hichiriki over wobble bass, ancient Japanese ritual in a dub chamber",
        "lyrics": "",
        "duration": 60,
        "bpm": 140,
        "keyscale": "C minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-highland-drone-trap",
        "caption": "Scottish highland bagpipe drone meets Atlanta trap, 808 bass under piobaireachd variations, mist over the plantation",
        "lyrics": "",
        "duration": 60,
        "bpm": 140,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-raag-afrobeats",
        "caption": "Indian classical raag meets West African afrobeats, sitar and tabla over shaker patterns and talking drum, Lagos meets Varanasi at sunset",
        "lyrics": "",
        "duration": 60,
        "bpm": 108,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-fado-hyperpop",
        "caption": "Portuguese fado meets hyperpop, acoustic guitar and saudoso vocal over glitchy production and pitched vocals, Lisbon in a digital funhouse",
        "lyrics": "",
        "duration": 60,
        "bpm": 120,
        "keyscale": "F minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT C: Prompt Detail Study Vol. 2 ═══════════
EXPERIMENT_C = [
    {
        "name": "sf21-medium-folk",
        "caption": "Warm indie folk with fingerpicked guitar and soft cello. Female vocal in an intimate room. Build quietly through two verses.",
        "lyrics": "",
        "duration": 60,
        "bpm": 85,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-medium-jazz",
        "caption": "Cool jazz trio with walking bass and brushed snare. Tenor sax takes the melody with restraint. Room sound is close and warm.",
        "lyrics": "",
        "duration": 60,
        "bpm": 120,
        "keyscale": "Bb major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf21-medium-electronic",
        "caption": "Deep house with warm analog pads and a round bassline. Subtle percussion builds over four minutes. The groove settles into a hypnotic pocket.",
        "lyrics": "",
        "duration": 60,
        "bpm": 122,
        "keyscale": "F minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: Seed Reproducibility Study ═══════════
SEED_CAPTION = "Fingerpicked acoustic guitar in A minor, gentle piano accents, warm cello drone, intimate room, autumn melancholy"
EXPERIMENT_D = [
    {
        "name": f"sf21-seed-{seed}",
        "caption": SEED_CAPTION,
        "lyrics": "",
        "duration": 60,
        "bpm": 80,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
        "seed": seed,
    }
    for seed in [42, 777, 2024]
]

# ═══════════ EXPERIMENT E: Guidance Scale Sweep ═══════════
GUIDANCE_CAPTION = "Sparse piano and cello, winter morning, snow falling outside, a single candle burning, deep silence between notes"
EXPERIMENT_E = [
    {
        "name": f"sf21-guidance-{g}",
        "caption": GUIDANCE_CAPTION,
        "lyrics": "",
        "duration": 60,
        "bpm": 70,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": g,
    }
    for g in [3.0, 5.0, 10.0, 15.0]
]

# ═══════════ EXPERIMENT F: Key Signature Study ═══════════
KEY_CAPTION = "Solo piano, gentle arpeggios, music box quality, nostalgic, the last light of an autumn afternoon through a window"
EXPERIMENT_F = [
    {
        "name": f"sf21-key-{key.replace(' ', '-').replace('#', 'sharp')}",
        "caption": KEY_CAPTION,
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": key,
        "inference_steps": 8,
        "guidance_scale": 7.0,
    }
    for key in ["C major", "E major", "Bb minor", "F# minor"]
]

ALL_TRACKS = EXPERIMENT_A + EXPERIMENT_B + EXPERIMENT_C + EXPERIMENT_D + EXPERIMENT_E + EXPERIMENT_F

def generate_track(handler, track):
    name = track["name"]
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Generating: {name}")
    logger.info(f"  Duration: {track['duration']}s | BPM: {track['bpm']} | Key: {track['keyscale']}")
    logger.info(f"  Steps: {track['inference_steps']} | Guidance: {track['guidance_scale']}")
    logger.info(f"  Caption: {track['caption'][:100]}...")
    logger.info(f"  Lyrics: {'yes' if track['lyrics'] else 'instrumental'} ({len(track['lyrics'])} chars)")

    gen_params = GenerationParams(
        caption=track["caption"],
        lyrics=track["lyrics"],
        duration=track["duration"],
        bpm=track["bpm"],
        keyscale=track["keyscale"],
        inference_steps=track["inference_steps"],
        guidance_scale=track["guidance_scale"],
        seed=track.get("seed", -1),
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

        # Find output file
        file_path = None
        file_size = 0
        if isinstance(result, dict) and "audio_path" in result:
            file_path = result["audio_path"]
        elif isinstance(result, list):
            for r in result:
                if isinstance(r, str) and os.path.exists(r):
                    file_path = r
                    break
        elif isinstance(result, str) and os.path.exists(result):
            file_path = result
        elif hasattr(result, "audio_path"):
            file_path = result.audio_path

        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  📁 File: {file_path} ({file_size:.2f}MB)")
        else:
            # Check for recently created files
            files = sorted(
                [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp3')],
                key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)),
                reverse=True
            )
            if files:
                file_path = os.path.join(SAVE_DIR, files[0])
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  📁 Most recent file: {file_path} ({file_size:.2f}MB)")

        return {
            "name": name,
            "elapsed": round(elapsed, 1),
            "size_mb": round(file_size, 2),
            "path": file_path,
            "duration": track["duration"],
            "bpm": track["bpm"],
            "keyscale": track["keyscale"],
            "guidance_scale": track["guidance_scale"],
            "lyrics": "yes" if track["lyrics"] else "instrumental",
            "seed": track.get("seed", -1),
            "status": "success",
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ❌ Failed after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return {
            "name": name,
            "elapsed": round(elapsed, 1),
            "error": str(e),
            "status": "failed",
        }

def main():
    logger.info("=" * 70)
    logger.info("  SongForge Session 21 — The Sunday Afternoon Laboratory")
    logger.info("=" * 70)
    logger.info(f"  A: Feedback Loop Vol.4 = {len(EXPERIMENT_A)}")
    logger.info(f"  B: Impossible Genres Vol.6 = {len(EXPERIMENT_B)}")
    logger.info(f"  C: Prompt Detail Vol.2 = {len(EXPERIMENT_C)}")
    logger.info(f"  D: Seed Reproducibility = {len(EXPERIMENT_D)}")
    logger.info(f"  E: Guidance Sweep = {len(EXPERIMENT_E)}")
    logger.info(f"  F: Key Signature Study = {len(EXPERIMENT_F)}")
    logger.info(f"  Total: {len(ALL_TRACKS)} tracks")
    logger.info("=" * 70)

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

    os.makedirs(SAVE_DIR, exist_ok=True)

    results = []
    for i, track in enumerate(ALL_TRACKS, 1):
        logger.info(f"\n{'═'*60}")
        logger.info(f"  TRACK {i}/{len(ALL_TRACKS)}")
        logger.info(f"{'═'*60}")
        result = generate_track(handler, track)
        results.append(result)

        # Save intermediate results
        results_path = os.path.join(SAVE_DIR, "session21_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Cleanup
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("  SESSION 21 COMPLETE — SUMMARY")
    logger.info("=" * 70)
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_time = sum(r["elapsed"] for r in results)
    total_size = sum(r.get("size_mb", 0) for r in results)

    logger.info(f"  Succeeded: {len(succeeded)}/{len(results)}")
    logger.info(f"  Failed: {len(failed)}/{len(results)}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"  Total size: {total_size:.2f}MB")

    for r in succeeded:
        logger.info(f"  ✅ {r['name']}: {r['elapsed']:.1f}s, {r['size_mb']:.2f}MB")
    for r in failed:
        logger.info(f"  ❌ {r['name']}: {r.get('error', 'unknown')[:80]}")

if __name__ == "__main__":
    main()
