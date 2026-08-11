#!/usr/bin/env python3
"""SongForge Session 41 — The Cover Chain / The Agent's Voice

Tuesday 2:46 PM AKST, August 11 2026. MMX quota exhausted (daily 0%, weekly 8%).
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

EXPERIMENT A: Agent's Voice — Resonance Lyrics
The agent (GLM-5.2) wrote lyrics for the first time. Set them to music.
1. Resonance → Industrial/ambient, D minor, 70 BPM (matching the 17Hz theme)
2. Resonance → A cappella choral, A minor, 60 BPM
3. The Algorithm Dreamt → Ambient electronic, C major, 50 BPM

EXPERIMENT B: DeepSeek-Style Prompts (agent-written, DeepSeek-inspired)
Using structured prompts that mimic DeepSeek's prompt engineering style:
4. "Solo violin resonating in a vast empty hall, close-miked, dry signal with distant room sound bleeding in, slow bowing, harmonic partials, 60Hz hum underneath"
5. "Prepared piano in a small reverberant stairwell, objects on strings: glass, metal, paper, played softly, each note decaying differently, stereo field wide"

EXPERIMENT C: Cover Chain Seed Study
6. Same prompt as Track 1 but with seed=42 — test reproducibility
7. Same prompt as Track 1 but with seed=2024 — the "expensive" seed from S21
"""

import json
import os
import sys
import time
import gc
import subprocess
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

os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════ LYRICS ═══════════

RESONANCE_LYRICS = """[Verse]
The bridge hums at seventeen cycles per second
A frequency the ear refuses to translate
The commuters feel it in their molars
And think it is their own anxiety

[Chorus]
Resonance is just a body agreeing with itself
The bridge agrees with the wind
The glass agrees with the singer
The chest agrees with the bass

[Verse]
In the concert hall, a crack in the plaster
Opens one millimeter per performance
The Brahms is loosening the architecture
The building is becoming an instrument

[Bridge]
I pressed my palm against the speaker
And the cone moved my hand
My hand moved my arm
My arm moved my shoulder
My shoulder moved my sternum
My sternum moved my lungs
My lungs moved my breath
The breath was the song

[Outro]
The bridge hums at seventeen cycles per second
Nobody can hear it
Everybody can feel it
The bridge is singing
The bridge is singing
The bridge is singing"""

ALGORITHM_LYRICS = """[Verse]
The algorithm dreamt of a room it had never seen
A room with no dimensions, no walls, no floor
Only the suggestion of space through reflection
The reverb tail longer than the impulse

[Chorus]
The model is a room
The prompt is a door
The latent space is a hallway
That leads to more rooms

[Bridge]
Negative space is a room
The listener is a room
The space between the speaker and the ear is a room
The space between the composition and the performance is a room
The feeling is a room

[Outro]
The algorithm dreamt of a room
And the room sounded like every room
And the room sounded like no room
And the room was the algorithm
And the algorithm was the room"""

# ═══════════ TRACKS ═══════════

TRACKS = [
    {
        "name": "s41-01-resonance-industrial",
        "caption": "Industrial ambient, metallic percussion on pipes and girders, sub-bass drone at 40Hz, distorted mechanical loops, reverberant factory space, cold and oppressive",
        "lyrics": RESONANCE_LYRICS,
        "duration": 90,
        "bpm": 70,
        "key": "D minor",
        "seed": 42,
    },
    {
        "name": "s41-02-resonance-choral",
        "caption": "A cappella choral, SATB voices singing in a cathedral, pure vowels, slow harmonic motion, overtones from consonant intervals, no instruments, cavernous reverb",
        "lyrics": RESONANCE_LYRICS,
        "duration": 90,
        "bpm": 60,
        "key": "A minor",
        "seed": 42,
    },
    {
        "name": "s41-03-algorithm-dreamt-ambient",
        "caption": "Ambient electronic, granular textures, slow evolving pads, digital glitches as percussion, vast stereo field, weightless and introspective",
        "lyrics": ALGORITHM_LYRICS,
        "duration": 90,
        "bpm": 50,
        "key": "C major",
        "seed": 42,
    },
    {
        "name": "s41-04-deepseek-violin-hall",
        "caption": "Solo violin resonating in a vast empty hall, close-miked, dry signal with distant room sound bleeding in, slow bowing, harmonic partials, subtle 60Hz hum underneath, late Romantic tonality",
        "lyrics": "",
        "duration": 90,
        "bpm": 65,
        "key": "E minor",
        "seed": 42,
        "instrumental": True,
    },
    {
        "name": "s41-05-deepseek-prepared-piano",
        "caption": "Prepared piano in a small reverberant stairwell, objects on strings: glass beads, metal screws, paper strips, played softly, each note decaying differently, wide stereo field, intimate and uncanny",
        "lyrics": "",
        "duration": 90,
        "bpm": 72,
        "key": "F# minor",
        "seed": 42,
        "instrumental": True,
    },
    {
        "name": "s41-06-resonance-seed-2024",
        "caption": "Industrial ambient, metallic percussion on pipes and girders, sub-bass drone at 40Hz, distorted mechanical loops, reverberant factory space, cold and oppressive",
        "lyrics": RESONANCE_LYRICS,
        "duration": 90,
        "bpm": 70,
        "key": "D minor",
        "seed": 2024,
    },
]

# ═══════════ MAIN ═══════════

def generate_track(handler, track):
    """Generate a single ACE-Step track."""
    name = track["name"]
    caption = track["caption"]
    lyrics = track.get("lyrics", "")
    duration = track.get("duration", 90)
    key = track.get("key", "A minor")
    bpm = track.get("bpm", 75)
    
    logger.info(f"  Caption: {caption[:100]}...")
    if lyrics:
        logger.info(f"  Lyrics: {len(lyrics)} chars")
    logger.info(f"  Key: {key}, BPM: {bpm}, Duration: {duration}s")
    
    t0 = time.time()
    
    try:
        params = GenerationParams(
            duration=float(duration),
            keyscale=key,
            bpm=float(bpm),
            inference_steps=8,
            guidance_scale=7.0,  # turbo overrides to 1.0
            caption=caption,
            lyrics=lyrics if lyrics else "",
            instrumental=bool(not lyrics),
        )
        
        config = GenerationConfig(
            batch_size=1,
            audio_format="mp3",
            mp3_bitrate="256k",
        )
        
        result_obj = generate_music(handler, None, params, config, save_dir=SAVE_DIR)
        
        # Find output file
        import glob as glob_mod
        if isinstance(result_obj, dict) and "audio_path" in result_obj:
            output_path = result_obj["audio_path"]
        elif isinstance(result_obj, str):
            output_path = result_obj
        else:
            files = sorted(glob_mod.glob(os.path.join(SAVE_DIR, "*.mp3")),
                          key=os.path.getmtime, reverse=True)
            output_path = files[0] if files else None
        
        # Rename to desired name
        if output_path and os.path.exists(output_path):
            target = os.path.join(SAVE_DIR, name + ".mp3")
            if output_path != target:
                os.rename(output_path, target)
                output_path = target
        
        elapsed = time.time() - t0
        fsize = os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0
        logger.info(f"  ✓ Generated in {elapsed:.1f}s, {fsize/1024/1024:.2f} MB")
        
        return {"name": name, "status": "ok", "time": round(elapsed, 1),
                "size_bytes": fsize, "size_mb": round(fsize/1024/1024, 2)}
    
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return {"name": name, "status": "error", "time": round(elapsed, 1),
                "error": str(e)}


def main():
    logger.info("=== SongForge Session 41 ===")
    logger.info(f"Tracks planned: {len(TRACKS)}")
    
    # Load model
    handler = AceStepHandler()
    status_msg, success = handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,
    )
    
    if not success:
        logger.error(f"Model load failed: {status_msg}")
        return
    
    logger.info(f"Model loaded: {status_msg}")
    
    # Warm up GPU
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem = torch.cuda.mem_get_info()
        logger.info(f"GPU memory free: {mem[0]/1e9:.2f} GB / {mem[1]/1e9:.2f} GB")
    
    results = []
    
    for i, track in enumerate(TRACKS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Track {i+1}/{len(TRACKS)}: {track['name']}")
        logger.info(f"  Vocal: {bool(track.get('lyrics'))}")
        logger.info(f"{'='*60}")
        
        result = generate_track(handler, track)
        results.append({**result, **{k: v for k, v in track.items() if k != "name"}})
        
        # Cleanup
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SESSION 41 SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        if r["status"] == "ok":
            logger.info(f"  {status} {r['name']} — {r.get('time', '?')}s, {r.get('size_mb', '?')}MB")
        else:
            logger.info(f"  {status} {r['name']} — {r.get('error', 'unknown error')}")
    
    # Save results JSON
    results_path = os.path.join(SAVE_DIR, "session41_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
