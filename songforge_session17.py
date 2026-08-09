#!/usr/bin/env python3
"""SongForge Session 17 — The Saturday Night Deep Structure

New experiments with ACE-Step 1.5 turbo on RTX 4050:
A: New corpus adaptations (Cadence Caller, Proof is Performance, Silence After)
B: Impossible genre matrix vol. 2 (Bebop Country, Gamelan Dub, Qing Dynasty Opera House)
C: DeepSeek-authored lyrics comparison (using pre-written lyrics with different voice)
D: 360-second duration frontier (six minutes — the outer edge)
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

# Load lyrics
CADENCE = load_lyrics("lyrics-the-cadence-caller-listens.txt")
PROOF = load_lyrics("lyrics-the-proof-is-the-performance.txt")
SILENCE = load_lyrics("lyrics-the-silence-after.txt")
SHELL = load_lyrics("lyrics-the-shell-merchant.txt")
INTERVAL = load_lyrics("lyrics-the-interval.txt")

# ═══════════ EXPERIMENT A: New Corpus Adaptations ═══════════
# Three new essays set to music — each exploring a different facet of musical silence
EXPERIMENT_A = [
    {
        "name": "sf17-cadence-caller-jazz",
        "caption": "Cool jazz, piano trio, walking bass, brushed drums, smoky late-night atmosphere, spacious and patient",
        "lyrics": CADENCE,
        "duration": 90,
        "bpm": 78,
        "keyscale": "F major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-proof-performance-math-rock",
        "caption": "Math rock meets show tune, angular guitar lines, odd time signatures, theatrical piano, dynamic shifts from intimate to explosive",
        "lyrics": PROOF,
        "duration": 90,
        "bpm": 97,
        "keyscale": "A major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-silence-after-ambient",
        "caption": "Post-classical ambient, solo piano with massive reverb, distant strings, tape hiss, the sound of a hall after the last note",
        "lyrics": SILENCE,
        "duration": 90,
        "bpm": 50,
        "keyscale": "D major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: Impossible Genre Matrix Vol. 2 ═══════════
# Push the fusion envelope further with more culturally distant combinations
EXPERIMENT_B = [
    {
        "name": "sf17-bebop-country",
        "caption": "Bebop country, fast walking bass with pedal steel, Coltrane changes on banjo, scat vocals over fiddle breaks, impossible swing",
        "lyrics": "",  # instrumental
        "duration": 60,
        "bpm": 160,
        "keyscale": "B-flat major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-gamelan-dub",
        "caption": "Indonesian gamelan meets Jamaican dub. Bronze gongs and metallophones over deep bass and echo, reverb tails on bell tones, King Tubby meets Bali",
        "lyrics": "",
        "duration": 60,
        "bpm": 68,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-peking-opera-trap",
        "caption": "Peking opera meets trap. Erhu and jinghu over 808 bass, stylized vocal cries, cymbals and hi-hats, ancient court drama in a modern cypher",
        "lyrics": "",
        "duration": 60,
        "bpm": 130,
        "keyscale": "F# minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-fado-techno",
        "caption": "Portuguese fado meets Berlin techno. Fado guitar and mournful female vocal over relentless four-on-the-floor, saudade on the dancefloor",
        "lyrics": SHELL,  # reuse Shell Merchant lyrics — they fit the melancholy
        "duration": 60,
        "bpm": 124,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT C: Duration Frontier — 360 seconds ═══════════
# Six full minutes — pushing past the 300s frontier from session 16
EXPERIMENT_C = [
    {
        "name": "sf17-duration-360-deep-ambient",
        "caption": "Six-minute deep ambient drift. Sub-bass at 28Hz, glacier-slow harmonic motion, occasional piano notes like distant lighthouses, the sound of tectonic plates having a conversation",
        "lyrics": "",
        "duration": 360,
        "bpm": 35,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf17-duration-360-cinematic",
        "caption": "Six-minute cinematic progression. Starts with solo cello, adds strings, builds to full orchestral moment, then decays back to silence. The arc of a film score in one continuous movement",
        "lyrics": "",
        "duration": 360,
        "bpm": 60,
        "keyscale": "A minor",
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
    if track["lyrics"]:
        logger.info(f"  Lyrics: {len(track['lyrics'])} chars")

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

        return elapsed, result
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ❌ Failed after {elapsed:.1f}s: {e}")
        return -1, None


def run_experiment_group(handler, label, title, tracks):
    logger.info(f"\n{'═'*60}")
    logger.info(f"EXPERIMENT {label}: {title}")
    logger.info(f"{'═'*60}")

    timings = {}
    for i, track in enumerate(tracks):
        elapsed, result = generate_track(handler, track)
        timings[track["name"]] = {
            "elapsed": elapsed,
            "duration": track["duration"],
            "guidance": track["guidance_scale"],
            "bpm": track["bpm"],
            "lyrics_chars": len(track["lyrics"]) if track["lyrics"] else 0,
        }
        # Cleanup between tracks
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        time.sleep(2)

    return timings


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_timings = {}

    logger.info(f"\n{'═'*60}")
    logger.info("Session 17: The Saturday Night Deep Structure")
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

    # A: New corpus adaptations
    all_timings["A_corpus"] = run_experiment_group(
        handler, "A", "New Corpus Adaptations — Silence Triptych", EXPERIMENT_A
    )

    # B: Impossible genres vol. 2
    all_timings["B_genres"] = run_experiment_group(
        handler, "B", "Impossible Genre Matrix Vol. 2", EXPERIMENT_B
    )

    # C: Duration frontier — 360s
    all_timings["C_duration"] = run_experiment_group(
        handler, "C", "Duration Frontier — 360 Seconds", EXPERIMENT_C
    )

    # Summary
    logger.info(f"\n{'═'*60}")
    logger.info("SESSION 17 SUMMARY")
    logger.info(f"{'═'*60}")
    total_tracks = 0
    total_time = 0
    for group, timings in all_timings.items():
        for name, data in timings.items():
            total_tracks += 1
            if data["elapsed"] > 0:
                total_time += data["elapsed"]
                logger.info(f"  {name}: {data['elapsed']:.1f}s gen ({data['duration']}s audio)")
            else:
                logger.info(f"  {name}: FAILED")
    logger.info(f"\nTotal: {total_tracks} tracks, {total_time:.0f}s generation time")

    # Save timings
    timings_path = os.path.join(SAVE_DIR, "sf17-timings.json")
    with open(timings_path, "w") as f:
        json.dump(all_timings, f, indent=2)
    logger.info(f"Timings saved to {timings_path}")


if __name__ == "__main__":
    main()
