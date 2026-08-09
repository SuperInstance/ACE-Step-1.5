#!/usr/bin/env python3
"""SongForge Session 18 — The Ouroboros Eats Its Eighth Tail

Saturday night, August 8 2026. MMX weekly quota exhausted (resets Aug 16).
ACE-Step 1.5 turbo on RTX 4050 Laptop GPU.

Tonight's experiments:
A: Essay-Music Feedback Loop — set the creative essays ABOUT the music back to music
B: DeepSeek/GLM as lyricist — comparison with existing M3 lyrics for the same concept
C: New impossible genres — klezmer drum & bass, Noh jazz, baroque dubstep
D: 420-second duration frontier — seven minutes, pushing past session 17's 360s
E: Guidance scale × vocals interaction — does guidance affect vocal tracks differently?
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
OUROBOROS = load_lyrics("lyrics-the-ouroboros-sings.txt")
INTERVAL_SINGS = load_lyrics("lyrics-the-interval-sings.txt")
HORIZON = load_lyrics("lyrics-the-six-minute-horizon.txt")
SHELL = load_lyrics("lyrics-the-shell-merchant.txt")
TAP = load_lyrics("lyrics-the-tap-sings.txt")

# ═══════════ EXPERIMENT A: Essay-Music Feedback Loop ═══════════
# The project eats its own tail: essays about songs become songs about essays about songs
EXPERIMENT_A = [
    {
        "name": "sf18-ouroboros-sings",
        "caption": "Ambient electronic with warm analog synths, recursive structure, looping melody that returns to its origin, deep sub-bass, patient and meditative",
        "lyrics": OUROBOROS,
        "duration": 90,
        "bpm": 70,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf18-interval-sings",
        "caption": "Cool jazz ambient, spacious trumpet with heavy reverb, slow piano chords, upright bass, the sound of distance made musical",
        "lyrics": INTERVAL_SINGS,
        "duration": 90,
        "bpm": 65,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf18-six-minute-horizon",
        "caption": "Deep ambient with cinematic build, sub-bass drone, orchestral strings entering gradually, patient evolution from silence to presence",
        "lyrics": HORIZON,
        "duration": 90,
        "bpm": 50,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: GLM vs M3 Lyricist Comparison ═══════════
# Same concept ("The Shell Merchant"), same prompt, same parameters
# but one uses M3-written lyrics (from session 4) and one uses GLM-written lyrics
# GLM lyrics are the agent's own (written above for the ouroboros)
# We reuse Shell Merchant (M3) and compare with the agent-written Interval Sings
# Both are about the same concept: the space between sounds
EXPERIMENT_B = [
    {
        "name": "sf18-lyricist-m3-shell-merchant",
        "caption": "Folk baroque, fingerpicked guitar, harpsichord, warm cello, intimate vocal",
        "lyrics": SHELL,  # M3-written, temp 0.92, session 4
        "duration": 90,
        "bpm": 72,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf18-lyricist-agent-interval",
        "caption": "Folk baroque, fingerpicked guitar, harpsichord, warm cello, intimate vocal",
        "lyrics": INTERVAL_SINGS,  # agent-written, session 18
        "duration": 90,
        "bpm": 72,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT C: New Impossible Genres ═══════════
# Continue pushing the genre fusion envelope
EXPERIMENT_C = [
    {
        "name": "sf18-klezmer-dnb",
        "caption": "Klezmer drum and bass, clarinet wailing over Reese bass and amen breaks, freylekh mode at 170 BPM, hora meets hyperdub",
        "lyrics": "",
        "duration": 60,
        "bpm": 170,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf18-noh-jazz",
        "caption": "Noh theater meets cool jazz, nohkan flute over walking bass, austere vocal styling, matsuri drums with brushed snare, ancient restraint meets blue note",
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": "F minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf18-baroque-dubstep",
        "caption": "Baroque dubstep, harpsichord and cello over wobble bass and half-time drums, Bach counterpoint with sub-bass drops, figured bass meets bass drop",
        "lyrics": "",
        "duration": 60,
        "bpm": 140,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: 420-Second Duration Frontier ═══════════
# Seven minutes — pushing past session 17's 360s
EXPERIMENT_D = [
    {
        "name": "sf18-duration-420-ambient",
        "caption": "Seven-minute deep ambient meditation. Sub-bass drone at 30Hz, glacier-slow harmonic movement, occasional bell tones, the sound of deep space breathing",
        "lyrics": "",
        "duration": 420,
        "bpm": 30,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT E: Guidance Scale × Vocals Interaction ═══════════
# Same vocal track at 3 different guidance scales
# Question: does guidance affect vocal clarity differently than instrumental clarity?
EXPERIMENT_E = [
    {
        "name": "sf18-guidance-vocals-050",
        "caption": "Atmospheric indie folk, fingerpicked guitar, haunting female vocal, sparse cello",
        "lyrics": TAP,
        "duration": 60,
        "bpm": 72,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 5.0,
    },
    {
        "name": "sf18-guidance-vocals-090",
        "caption": "Atmospheric indie folk, fingerpicked guitar, haunting female vocal, sparse cello",
        "lyrics": TAP,
        "duration": 60,
        "bpm": 72,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 9.0,
    },
    {
        "name": "sf18-guidance-vocals-150",
        "caption": "Atmospheric indie folk, fingerpicked guitar, haunting female vocal, sparse cello",
        "lyrics": TAP,
        "duration": 60,
        "bpm": 72,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 15.0,
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
        import traceback
        traceback.print_exc()
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
    logger.info("Session 18: The Ouroboros Eats Its Eighth Tail")
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

    # A: Essay-music feedback loop
    all_timings["A_feedback_loop"] = run_experiment_group(
        handler, "A", "Essay-Music Feedback Loop — The Ouroboros", EXPERIMENT_A
    )

    # B: Lyricist comparison
    all_timings["B_lyricist"] = run_experiment_group(
        handler, "B", "Lyricist Comparison — M3 vs Agent", EXPERIMENT_B
    )

    # C: New impossible genres
    all_timings["C_genres"] = run_experiment_group(
        handler, "C", "Impossible Genre Matrix Vol. 3", EXPERIMENT_C
    )

    # E: Guidance × vocals (run before D so we don't end on a 7-minute track)
    all_timings["E_guidance_vocals"] = run_experiment_group(
        handler, "E", "Guidance Scale × Vocals Interaction", EXPERIMENT_E
    )

    # D: 420s duration frontier (run LAST — it takes ~10 minutes)
    all_timings["D_duration"] = run_experiment_group(
        handler, "D", "Duration Frontier — 420 Seconds (7 Minutes)", EXPERIMENT_D
    )

    # Summary
    logger.info(f"\n{'═'*60}")
    logger.info("SESSION 18 SUMMARY")
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
    timings_path = os.path.join(SAVE_DIR, "sf18-timings.json")
    with open(timings_path, "w") as f:
        json.dump(all_timings, f, indent=2)
    logger.info(f"Timings saved to {timings_path}")


if __name__ == "__main__":
    main()
