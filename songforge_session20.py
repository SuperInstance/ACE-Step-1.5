#!/usr/bin/env python3
"""SongForge Session 20 — The Sunday Long Breath (Part 2)

Sunday afternoon. MMX weekly quota exhausted (resets ~16:00 AKST today).
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

Five frontiers:
A: Essay-Music Feedback Loop Vol. 3 — Session 19 fictions become songs
B: Tempo Threshold Study — 40, 50, 60, 70 BPM to find where diffusion cost spikes
C: Impossible Genre Matrix Vol. 5 — new culturally distant fusions
D: Prompt Detail Study Vol. 2 — confirming the medium-prompt sweet spot
E: Duration — 480s Cinematic with "warm kernel" prompt
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

# Load lyrics for feedback loop tracks
BLUEGRASS = load_lyrics("lyrics-the-bluegrass-spring-reverb.txt")
PANSORI = load_lyrics("lyrics-the-pansori-feedback.txt")
DAKAR = load_lyrics("lyrics-dakar-berlin-128.txt")
SUNDAY = load_lyrics("lyrics-the-sunday-long-breath.txt")

# ═══════════ EXPERIMENT A: Essay-Music Feedback Loop Vol. 3 ═══════════
EXPERIMENT_A = [
    {
        "name": "sf20-bluegrass-spring-reverb",
        "caption": "Bluegrass meets dub, banjo and dobro over deep bass with spring reverb delays, Appalachian holler soundsystem",
        "lyrics": BLUEGRASS,
        "duration": 90,
        "bpm": 88,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-pansori-feedback",
        "caption": "Pansori meets grunge, Korean traditional vocal over distorted guitar and buk drum, feedback as ancient mourning",
        "lyrics": PANSORI,
        "duration": 90,
        "bpm": 95,
        "keyscale": "B minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-dakar-berlin-techno",
        "caption": "Mbalax techno, Sabar drums over four-on-the-floor kick, talking drum, Dakar meets Berlin at 128 BPM",
        "lyrics": DAKAR,
        "duration": 90,
        "bpm": 128,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: Tempo Threshold Study ═══════════
BASE_CAPTION_B = "Fingerpicked acoustic guitar, warm cello, gentle piano, intimate room recording, autumn afternoon melancholy"
EXPERIMENT_B = [
    {
        "name": f"sf20-tempo-{bpm}",
        "caption": BASE_CAPTION_B,
        "lyrics": "",
        "duration": 90,
        "bpm": bpm,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    }
    for bpm in [40, 50, 60, 70]
]

# ═══════════ EXPERIMENT C: Impossible Genre Matrix Vol. 5 ═══════════
EXPERIMENT_C = [
    {
        "name": "sf20-polka-black-metal",
        "caption": "Polka meets Norwegian black metal, accordion and tremolo guitar, blast beats with oompah bass, folk horror wedding",
        "lyrics": "",
        "duration": 60,
        "bpm": 140,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-zydeco-shoegaze",
        "caption": "Zydeco meets shoegaze, accordion and washboard buried in walls of guitar fuzz, Cajun feedback, Louisiana dreamscape",
        "lyrics": "",
        "duration": 60,
        "bpm": 120,
        "keyscale": "A major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-flamenco-dnb",
        "caption": "Flamenco meets drum and bass, palmas and cajon over breakbeats at 170 BPM, Spanish guitar through Reece bass",
        "lyrics": "",
        "duration": 60,
        "bpm": 170,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-mariachi-synthwave",
        "caption": "Mariachi meets synthwave, trumpet and guitarron over analog synths and gated reverb, neon serenade, Retrowave Guadalajara",
        "lyrics": "",
        "duration": 60,
        "bpm": 110,
        "keyscale": "C minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: Prompt Detail Study Vol. 2 ═══════════
EXPERIMENT_D = [
    {
        "name": "sf20-prompt-medium-folk",
        "caption": "Three sentences of direction. Warm indie folk with fingerpicked guitar and soft cello. Female vocal in an intimate room. Build quietly through two verses.",
        "lyrics": SUNDAY,
        "duration": 90,
        "bpm": 72,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-prompt-medium-jazz",
        "caption": "Three sentences of direction. Late night cool jazz, piano trio with brushes. Walking bass and smoky atmosphere. Let the space between notes breathe.",
        "lyrics": SUNDAY,
        "duration": 90,
        "bpm": 78,
        "keyscale": "F major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf20-prompt-medium-electronic",
        "caption": "Three sentences of direction. Gentle ambient electronic with warm pads and soft pulse. Sub-bass underneath, sparkle on top. Slowly evolving texture.",
        "lyrics": SUNDAY,
        "duration": 90,
        "bpm": 65,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT E: Duration — 480s Warm Cinematic ═══════════
EXPERIMENT_E = [
    {
        "name": "sf20-duration-480-warm-cinematic",
        "caption": "Eight minute journey. Solo piano begins, cello enters at minute two, full string section by minute four, decays to piano again by minute six, silence at eight.",
        "lyrics": "",
        "duration": 480,
        "bpm": 55,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

ALL_TRACKS = EXPERIMENT_A + EXPERIMENT_B + EXPERIMENT_C + EXPERIMENT_D + EXPERIMENT_E

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

        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  📁 File: {file_path} ({file_size:.2f}MB)")
        else:
            # Check for any recently created file
            logger.info(f"  ⚠️ Could not find output file. Result type: {type(result)}")

        return {
            "name": name,
            "elapsed": round(elapsed, 1),
            "size_mb": round(file_size, 2),
            "path": file_path,
            "duration": track["duration"],
            "bpm": track["bpm"],
            "keyscale": track["keyscale"],
            "lyrics": "yes" if track["lyrics"] else "instrumental",
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
    logger.info("SongForge Session 20 — The Sunday Long Breath (Part 2)")
    logger.info(f"Total tracks planned: {len(ALL_TRACKS)}")
    logger.info(f"  A: Feedback Loop Vol.3 = {len(EXPERIMENT_A)}")
    logger.info(f"  B: Tempo Threshold = {len(EXPERIMENT_B)}")
    logger.info(f"  C: Impossible Genres Vol.5 = {len(EXPERIMENT_C)}")
    logger.info(f"  D: Prompt Detail Vol.2 = {len(EXPERIMENT_D)}")
    logger.info(f"  E: Duration 480 Warm = {len(EXPERIMENT_E)}")
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

    results = []
    for i, track in enumerate(ALL_TRACKS, 1):
        logger.info(f"\n{'═'*60}")
        logger.info(f"TRACK {i}/{len(ALL_TRACKS)}")
        result = generate_track(handler, track)
        results.append(result)

        # Save intermediate results
        results_path = os.path.join(SAVE_DIR, "session20_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Cleanup
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("SESSION 20 COMPLETE — SUMMARY")
    logger.info("=" * 70)
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_time = sum(r["elapsed"] for r in results)
    total_size = sum(r.get("size_mb", 0) for r in results)

    logger.info(f"Succeeded: {len(succeeded)}/{len(results)}")
    logger.info(f"Failed: {len(failed)}/{len(results)}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Total size: {total_size:.2f}MB")

    for r in succeeded:
        logger.info(f"  ✅ {r['name']}: {r['elapsed']:.1f}s, {r['size_mb']:.2f}MB")
    for r in failed:
        logger.info(f"  ❌ {r['name']}: {r.get('error', 'unknown')[:80]}")

if __name__ == "__main__":
    main()
