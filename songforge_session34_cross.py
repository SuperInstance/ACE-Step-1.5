#!/usr/bin/env python3
"""SongForge Session 34 — ACE-Step Cross-System Comparison

Monday 4:46 PM AKST, August 10 2026. MMX interval quota exhausted.
Weekly quota at 39% but interval gate (0%) blocks all MMX calls.
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

EXPERIMENT: Cross-System Synesthetic Prompt Comparison
Generate the SAME synesthetic prompts on ACE-Step that were generated on MMX.
This enables A/B comparison between the two systems for non-conventional prompts.

Questions:
1. Does ACE-Step's output size also increase for negative-space/synesthetic prompts?
2. Does ACE-Step interpret spatial metaphor differently than MMX?
3. How do the two systems compare for prompt-following on abstract descriptions?
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

# ═══════════ EXPERIMENT: Cross-System Synesthetic Prompts ═══════════
# Same prompts as MMX Session 34, adapted for ACE-Step's caption format
# ACE-Step uses a single caption string, not structured flags

TRACKS = [
    {
        "name": "sf34-ace-cavern-ocean",
        "caption": "A cavern that remembers being an ocean. Wet stone resonance humming beneath crystalline droplets that fall in 7/8 time, each impact refracting into harmonic galaxies that dissolve before they fully form. Deep ambient with metallic percussion and vast reverb.",
        "lyrics": "",
        "duration": 90,
        "key": "D minor",
        "bpm": 70,
    },
    {
        "name": "sf34-ace-frozen-waterfall",
        "caption": "Sunlight bending through a frozen waterfall, captured mid-collapse. Glass shards of melody suspended on thermoelectric drones while sub-bass pulses like tectonic plates arguing in slow motion. Ambient electronic with crystalline textures.",
        "lyrics": "",
        "duration": 90,
        "key": "A minor",
        "bpm": 60,
    },
    {
        "name": "sf34-ace-indigo-droplets",
        "caption": "Cool indigo droplets spill downward like condensation racing across glass, each impact a tiny silver spark that prickles the skin before dissolving into a thick velvet hum. Ambient with deep bass, metallic clicks, and reverb.",
        "lyrics": "",
        "duration": 90,
        "key": "F major",
        "bpm": 65,
    },
    {
        "name": "sf34-ace-magnet-silence",
        "caption": "The silence between two magnets. A held breath of tone so pure it aches, broken by friction-bowed strings that sing like wind through telephone wires in a city that hasn't been built yet. Drone with sustained strings and eerie harmonics.",
        "lyrics": "",
        "duration": 90,
        "key": "E minor",
        "bpm": 55,
    },
    {
        "name": "sf34-ace-neg-space-no-drums",
        "caption": "A pop song with absolutely no percussion. Only melody and harmony carrying the rhythm. Piano, strings, and voice only. The absence of drums is the statement. Gentle, flowing, introspective.",
        "lyrics": """[Verse]
In the space where the drum used to be
I found a silence shaped like a key
The piano plays the heartbeat now
The strings describe what the bass allowed

[Chorus]
No kick no snare no hat no beat
Just the melody complete
The rhythm lives inside the chord
The absence is its own reward""",
        "duration": 90,
        "key": "F major",
        "bpm": 90,
    },
    {
        "name": "sf34-ace-silence-instrument",
        "caption": "A minimalist composition where silence is the primary instrument. Long pauses between phrases. Each note appears alone in a vast empty space. The rests are more important than the notes. John Cage meets Arvo Part. Solo piano with massive reverb.",
        "lyrics": """[Verse]
One note
Then nothing
Then one note again
The space between
Is where I live

[Chorus]
The silence plays
The silence sings
The silence is
The loudest thing""",
        "duration": 90,
        "key": "B minor",
        "bpm": 60,
    },
]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize handler
    logger.info("Initializing ACE-Step handler...")
    t0 = time.time()

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

    logger.info(f"Handler initialized in {time.time()-t0:.1f}s. Status: {status_msg}")

    results = []

    for i, track in enumerate(TRACKS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Track {i+1}/{len(TRACKS)}: {track['name']}")
        logger.info(f"  Key: {track['key']}, BPM: {track['bpm']}, Duration: {track['duration']}s")
        logger.info(f"  Caption: {track['caption'][:80]}...")

        t0 = time.time()

        try:
            params = GenerationParams(
                duration=float(track["duration"]),
                keyscale=track["key"],
                bpm=track["bpm"],
                inference_steps=8,
                guidance_scale=7.0,  # turbo will override to 1.0
                caption=track["caption"],
                lyrics=track["lyrics"],
            )

            config = GenerationConfig(
                batch_size=1,
                audio_format="mp3",
                mp3_bitrate="256k",
            )

            result_obj = generate_music(handler, None, params, config, save_dir=SAVE_DIR)

            # Find output file
            if isinstance(result_obj, dict) and "audio_path" in result_obj:
                output_path = result_obj["audio_path"]
            elif isinstance(result_obj, str):
                output_path = result_obj
            else:
                import glob
                files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.mp3")), key=os.path.getmtime, reverse=True)
                output_path = files[0] if files else None

            # Rename to desired name
            if output_path and os.path.exists(output_path):
                target = os.path.join(SAVE_DIR, track["name"] + ".mp3")
                if output_path != target:
                    os.rename(output_path, target)
                    output_path = target

            elapsed = time.time() - t0
            fsize = os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0

            logger.info(f"  ✓ Generated in {elapsed:.1f}s, size: {fsize:,} bytes ({fsize/1024/1024:.2f} MB)")

            results.append({
                "name": track["name"],
                "duration": track["duration"],
                "size_bytes": fsize,
                "gen_time": elapsed,
                "status": "success",
            })

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  ✗ FAILED in {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append({
                "name": track["name"],
                "duration": track["duration"],
                "size_bytes": 0,
                "gen_time": elapsed,
                "status": f"failed: {e}",
            })

        # Cleanup
        gc.collect()
        if torch := __import__('torch'):
            torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("=== Session 34 ACE-Step Cross-System Summary ===")
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        size_mb = r["size_bytes"] / 1024 / 1024 if r["size_bytes"] else 0
        logger.info(f"  {status} {r['name']}: {size_mb:.2f} MB, {r['gen_time']:.1f}s")

    # Save results JSON
    results_path = os.path.join(SAVE_DIR, "sf34-ace-results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
