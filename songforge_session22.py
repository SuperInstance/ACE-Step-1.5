#!/usr/bin/env python3
"""SongForge Session 22 — "The Temporal Mismatch and the Breathing Room"

Sunday 2:46 PM AKST, August 9 2026. MMX quota still exhausted (weekly).
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload). 17GB RAM available.

Five experiments:
A: Feedback Loop Vol. 5 — Session 21 fictions become songs (4 tracks, 90s vocal)
B: Temporal Mismatch Study — prompt describes wrong duration (5 tracks, 60s instrumental)
C: Extreme Temperature Study — varying inference_steps dramatically (4 tracks, 60s instrumental)
D: The Breathing Room — prompts about breath and space across genres (3 tracks, 60s instrumental)
E: The Cover Project — Casey's lyrics in radically different styles (2 tracks, 90s vocal)
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

# Load lyrics for feedback loop vol.5
SHO_TIDE = load_lyrics("lyrics-the-sho-and-the-tide.txt")
DRONE_TRAP = load_lyrics("lyrics-the-drone-and-the-trap.txt")
SHAKER_RAAG = load_lyrics("lyrics-the-shaker-follows-the-raag.txt")
FUNHOUSE = load_lyrics("lyrics-the-funhouse-mirror.txt")

# Load Casey's original lyrics for cover experiments
CASEY = load_lyrics("/home/eileen/projects/ACE-Step-1.5/casey_lyrics.txt" if os.path.exists("/home/eileen/projects/ACE-Step-1.5/casey_lyrics.txt") else "../ACE-Step-1.5/casey_lyrics.txt")
# fallback
if not CASEY or len(CASEY) < 50:
    with open(os.path.join(PROJECT_ROOT, "casey_lyrics.txt")) as f:
        CASEY = f.read().strip()

# ═══════════ EXPERIMENT A: Feedback Loop Vol. 5 ═══════════
# Session 21 fiction → Session 22 lyrics → Session 22 music
EXPERIMENT_A = [
    {
        "name": "sf22-sho-and-the-tide",
        "caption": "Gagaku meets dubstep, sho drone and wobble bass, Japanese court music through Croydon bass bins, 140 BPM, circular breathing meets LFO, temple bells and sub bass",
        "lyrics": SHO_TIDE,
        "duration": 90,
        "bpm": 140,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-drone-and-the-trap",
        "caption": "Highland bagpipe meets Atlanta trap, bass drone and 808 kicks, Glenfinnan to Georgia, Bb drone at 470 Hz, pipe band marching through a trap beat, lament meets flex",
        "lyrics": DRONE_TRAP,
        "duration": 90,
        "bpm": 130,
        "keyscale": "Bb major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-shaker-follows-the-raag",
        "caption": "Raag Yaman meets afrobeats, sitar and shaker, evening raag in Lagos, polyrhythmic, three against four, Indian classical meets West African pop, sunset devotion",
        "lyrics": SHAKER_RAAG,
        "duration": 90,
        "bpm": 108,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-funhouse-mirror",
        "caption": "Fado meets hyperpop, Portuguese saudade through pitch correction, Berlin factory beats with Lisbon sadness, glitchy production, breaking voice, cracked perfection",
        "lyrics": FUNHOUSE,
        "duration": 90,
        "bpm": 120,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT B: Temporal Mismatch Study ═══════════
# Same prompt but with WRONG duration descriptions → measuring diffusion cost
MISMATCH_BASE = "Ambient electronic, warm pad textures, evolving soundscape, gentle arpeggios"
EXPERIMENT_B = [
    {
        "name": "sf22-temporal-30s",
        "caption": f"{MISMATCH_BASE}, a brief thirty-second moment of sound",
        "lyrics": "",
        "duration": 60,
        "bpm": 90,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-temporal-60s",
        "caption": f"{MISMATCH_BASE}, exactly one minute of continuous music",
        "lyrics": "",
        "duration": 60,
        "bpm": 90,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-temporal-2min",
        "caption": f"{MISMATCH_BASE}, building over two minutes",
        "lyrics": "",
        "duration": 60,
        "bpm": 90,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-temporal-4min",
        "caption": f"{MISMATCH_BASE}, building over four minutes",
        "lyrics": "",
        "duration": 60,
        "bpm": 90,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-temporal-10min",
        "caption": f"{MISMATCH_BASE}, an epic ten-minute journey through sound",
        "lyrics": "",
        "duration": 60,
        "bpm": 90,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT C: Extreme Inference Steps ═══════════
# Testing turbo model with extreme step counts
EXPERIMENT_C = [
    {
        "name": "sf22-steps-4",
        "caption": "Warm jazz piano trio, brushed drums, upright bass, smoky lounge, late night, melancholic swing",
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": "F major",
        "inference_steps": 4,  # Minimal — ultra-fast
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-steps-6",
        "caption": "Warm jazz piano trio, brushed drums, upright bass, smoky lounge, late night, melancholic swing",
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": "F major",
        "inference_steps": 6,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-steps-12",
        "caption": "Warm jazz piano trio, brushed drums, upright bass, smoky lounge, late night, melancholic swing",
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": "F major",
        "inference_steps": 12,  # More than default 8
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-steps-20",
        "caption": "Warm jazz piano trio, brushed drums, upright bass, smoky lounge, late night, melancholic swing",
        "lyrics": "",
        "duration": 60,
        "bpm": 75,
        "keyscale": "F major",
        "inference_steps": 20,  # Heavy — max quality?
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT D: The Breathing Room ══════════════
# Prompts about breath, space, and sustained sound across genres
EXPERIMENT_D = [
    {
        "name": "sf22-breath-ambient",
        "caption": "Deep ambient drone, single sustained note growing from silence, Tibetan singing bowl and cathedral reverb, the sound of breathing in an empty room, meditation, stillness",
        "lyrics": "",
        "duration": 60,
        "bpm": 40,
        "keyscale": "C major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-breath-classical",
        "caption": "Minimalist classical, Arvo Part tintinnabuli style, single piano note sustained and decaying, violin drone, silence as instrument, sacred music for empty cathedral, frozen chord",
        "lyrics": "",
        "duration": 60,
        "bpm": 50,
        "keyscale": "A minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-breath-dubtechno",
        "caption": "Dub techno, Basic Channel style, deep chord stabs through infinite delay, Detroit to Berlin, vinyl crackle, the groove is in the space between beats, submarine sonar bass",
        "lyrics": "",
        "duration": 60,
        "bpm": 120,
        "keyscale": "E minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

# ═══════════ EXPERIMENT E: Casey Cover Project ═════════════
# Casey's original lyrics in radically different genre treatments
EXPERIMENT_E = [
    {
        "name": "sf22-casey-doomjazz",
        "caption": "Dark doom jazz, slow blackened saxophone, funeral organ, doom metal guitar drone, Bohren und der Club of Gore meets Sunn O))), cinematic horror jazz, 40 BPM, suffocating and beautiful",
        "lyrics": CASEY,
        "duration": 90,
        "bpm": 40,
        "keyscale": "D minor",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
    {
        "name": "sf22-casey-bossa",
        "caption": "Warm bossa nova, gentle nylon guitar, soft brushed snares, female vocal in Portuguese style, Stan Getz meets Joao Gilberto, Sunday morning, cafe by the beach, warm and tender",
        "lyrics": CASEY,
        "duration": 90,
        "bpm": 110,
        "keyscale": "G major",
        "inference_steps": 8,
        "guidance_scale": 7.0,
    },
]

ALL_EXPERIMENTS = {
    "A_feedback_loop_5": EXPERIMENT_A,
    "B_temporal_mismatch": EXPERIMENT_B,
    "C_inference_steps": EXPERIMENT_C,
    "D_breathing_room": EXPERIMENT_D,
    "E_casey_covers": EXPERIMENT_E,
}

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timings = []
    handler = None

    # Determine which experiments to skip (already completed)
    completed = set()
    for fname in os.listdir(SAVE_DIR):
        if fname.startswith("sf22-") and fname.endswith(".mp3"):
            completed.add(fname.replace(".mp3", ""))
    logger.info(f"Already completed: {sorted(completed)}")

    pending = []
    for exp_name, tracks in ALL_EXPERIMENTS.items():
        for t in tracks:
            if t["name"] not in completed:
                pending.append((exp_name, t))

    if not pending:
        logger.info("All experiments already completed!")
        timings_path = os.path.join(SAVE_DIR, "sf22-timings.json")
        with open(timings_path, "w") as f:
            json.dump(timings, f, indent=2)
        return

    logger.info(f"Pending experiments: {len(pending)} tracks to generate")

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

    for exp_name, track in pending:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_name} | Track: {track['name']}")
        logger.info(f"  prompt: {track['caption'][:80]}...")
        logger.info(f"  duration: {track['duration']}s, bpm: {track['bpm']}, key: {track['keyscale']}")
        logger.info(f"  steps: {track['inference_steps']}, guidance: {track['guidance_scale']}")

        t0 = time.time()
        try:
            params = GenerationParams(
                caption=track["caption"],
                lyrics=track["lyrics"] if track["lyrics"] else "",
                duration=track["duration"],
                bpm=track["bpm"],
                keyscale=track["keyscale"],
                inference_steps=track["inference_steps"],
                guidance_scale=track["guidance_scale"],
                seed=42,
            )
            config = GenerationConfig(
                batch_size=1,
                audio_format="mp3",
                mp3_bitrate="256k",
            )

            result = generate_music(
                handler,
                None,
                params,
                config,
                save_dir=SAVE_DIR,
            )

            # Find output file
            output_path = None
            if isinstance(result, dict) and "audio_path" in result:
                output_path = result["audio_path"]
            elif isinstance(result, list):
                for r in result:
                    if isinstance(r, str) and os.path.exists(r):
                        output_path = r
                        break
            elif isinstance(result, str) and os.path.exists(result):
                output_path = result

            # Fallback: check for recent files
            if not output_path or not os.path.exists(output_path):
                recent = sorted(
                    [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp3')],
                    key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)),
                    reverse=True
                )
                if recent:
                    output_path = os.path.join(SAVE_DIR, recent[0])

            t1 = time.time()
            elapsed = t1 - t0
            file_size = os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0

            timing_entry = {
                "experiment": exp_name,
                "name": track["name"],
                "caption": track["caption"],
                "duration": track["duration"],
                "bpm": track["bpm"],
                "keyscale": track["keyscale"],
                "inference_steps": track["inference_steps"],
                "guidance_scale": track["guidance_scale"],
                "elapsed_seconds": round(elapsed, 2),
                "file_size_bytes": file_size,
                "output_path": output_path,
                "status": "success",
            }
            timings.append(timing_entry)
            logger.info(f"  ✓ Done in {elapsed:.2f}s, file size: {file_size} bytes")

            # Save timings after each track
            timings_path = os.path.join(SAVE_DIR, "sf22-timings.json")
            with open(timings_path, "w") as f:
                json.dump(timings, f, indent=2)

        except Exception as e:
            t1 = time.time()
            elapsed = t1 - t0
            timing_entry = {
                "experiment": exp_name,
                "name": track["name"],
                "elapsed_seconds": round(elapsed, 2),
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()[:500],
            }
            timings.append(timing_entry)
            logger.error(f"  ✗ FAILED in {elapsed:.2f}s: {e}")

            # Save timings
            timings_path = os.path.join(SAVE_DIR, "sf22-timings.json")
            with open(timings_path, "w") as f:
                json.dump(timings, f, indent=2)

        # Cleanup between tracks
        gc.collect()
        if hasattr(handler, 'model'):
            try:
                torch = __import__('torch')
                torch.cuda.empty_cache()
            except:
                pass

    logger.info(f"\n{'='*60}")
    logger.info(f"Session 22 complete. {len([t for t in timings if t.get('status') == 'success'])} succeeded, {len([t for t in timings if t.get('status') == 'failed'])} failed.")
    logger.info(f"Timings saved to {os.path.join(SAVE_DIR, 'sf22-timings.json')}")

if __name__ == "__main__":
    main()
