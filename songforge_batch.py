#!/usr/bin/env python3
"""SongForge Batch Cover Generation — Session 6

Generates multiple cover variants of Casey's "One Day" using ACE-Step 1.5.
Each variant uses a different style prompt while keeping Casey's original as reference.
Also generates text2music variants with Casey's lyrics.

Runs entirely locally on RTX 4050 (6GB VRAM) with CPU offloading.
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
ORIGINAL_AUDIO = "/home/eileen/projects/covers/onedayine.mp3"
LYRICS_FILE = "/home/eileen/projects/covers/casey_lyrics.txt"
SAVE_DIR = "/home/eileen/projects/covers/experiments_v4"

# Read lyrics
with open(LYRICS_FILE, "r") as f:
    CASEY_LYRICS = f.read().strip()

# ─── Cover Variants ───
# Each uses Casey's original as reference_audio with different style prompts
COVER_VARIANTS = [
    {
        "name": "v6_nashville_confession",
        "caption": (
            "Alternative country-folk in the vein of Jason Isbell crossed with Iron & Wine. "
            "A weathered male voice sings over fingerpicked acoustic guitar with warm tube amp character. "
            "Pedal steel guitar enters on the chorus, weeping quietly. "
            "Brushed snare and upright bass provide gentle rhythm. "
            "The production is warm and close-miked, like sitting in a small room with the musician. "
            "The singer has lived with these words for decades and finds new weight in them now."
        ),
        "bpm": 85,
        "keyscale": "E major",
        "audio_cover_strength": 0.8,
        "cover_noise_strength": 0.3,
    },
    {
        "name": "v6_3am_kitchen",
        "caption": (
            "Lo-fi bedroom folk recorded on a four-track cassette. "
            "The sound is intimate to the point of discomfort — you can hear the guitarist's fingers on the strings. "
            "Double-tracked vocals are slightly out of phase, creating a ghostly unison. "
            "A distant piano appears in the second verse, barely audible. "
            "The voice is a breathy male tenor, half-whispered, as if singing after everyone else has gone to sleep. "
            "No drums. No bass. Just the skeleton of the song exposed."
        ),
        "bpm": 72,
        "keyscale": "E major",
        "audio_cover_strength": 0.6,
        "cover_noise_strength": 0.5,
    },
    {
        "name": "v6_chamber_meditation",
        "caption": (
            "Chamber folk with the intimacy of Nick Drake and the orchestral ambition of Sufjan Stevens. "
            "A classical nylon-string guitar fingerpicks in E major. "
            "A cello holds a drone underneath. The male voice is whispered, close-miked. "
            "Piano notes fall like raindrops in the second verse. "
            "The chorus introduces a string quartet playing slow, suspensive chords. "
            "There are no drums anywhere. The song ends with a single guitar harmonic ringing into silence."
        ),
        "bpm": 65,
        "keyscale": "E major",
        "audio_cover_strength": 0.5,
        "cover_noise_strength": 0.6,
    },
]

# ─── Text2Music Variants ───
# These generate from scratch with Casey's lyrics, no reference audio
TEXT2MUSIC_VARIANTS = [
    {
        "name": "v6_gospel_hymn",
        "caption": (
            "Gospel-inflected folk rock that builds from a single voice and acoustic guitar to a full choir. "
            "The opening verse is solo — just a clear male tenor with a slight rasp, singing over fingerpicked guitar. "
            "By the first chorus, hand claps and a stomp board join. "
            "The second verse adds an upright bass. The bridge strips everything away. "
            "Then the final chorus explodes with a full gospel choir singing harmonies, a Hammond organ swelling underneath. "
            "It sounds like a revival tent in the deep south."
        ),
        "bpm": 100,
        "keyscale": "E major",
    },
    {
        "name": "v6_celtic_ballad",
        "caption": (
            "Traditional Celtic folk ballad adapted to contemporary indie production. "
            "Dropped-D acoustic guitar fingerpicking in E major, with uilleann pipes threading through the arrangement. "
            "A bodhrán frame drum provides a heartbeat rhythm on the choruses. "
            "The male voice is clear and unaffected, with a slight Irish inflection on elongated vowels. "
            "The production avoids modern sheen; everything sounds like it was recorded in a stone church. "
            "Fiddle enters on the final chorus, playing counter-melodies around the vocal line."
        ),
        "bpm": 80,
        "keyscale": "E major",
    },
    {
        "name": "v6_blues_crossroads",
        "caption": (
            "Delta blues meets indie folk. Open-E tuned acoustic guitar with a slide. "
            "The male voice is a gritty baritone that can break into a falsetto on emotional peaks. "
            "A foot stomp on a wooden floor provides the only percussion. "
            "Harmonica accents punctuate the choruses, played through a cupped mic. "
            "The production is bone-dry — no reverb, no delay, just the raw sound of a person and their instrument on a porch."
        ),
        "bpm": 90,
        "keyscale": "E major",
    },
    {
        "name": "v6_ambient_dreamscape",
        "caption": (
            "Ambient folk soundscapes inspired by Bon Iver's 'For Emma, Forever Ago' filtered through Sigur Rós. "
            "Acoustic guitar is processed through reverse reverbs and granular delays. "
            "The male vocal is primarily in falsetto, treated with subtle auto-tune for an otherworldly quality. "
            "Synth pads swell underneath, creating a bed of sound rather than a traditional arrangement. "
            "The choruses layer five or six vocal tracks, all the same singer, creating a choir of one."
        ),
        "bpm": 60,
        "keyscale": "E major",
    },
]


def init_handlers():
    """Initialize DiT and LLM handlers."""
    logger.info("Initializing DiT handler (turbo)...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=True,  # Required for 6GB VRAM
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s — {status_msg}")

    # Skip LM for VRAM savings — turbo mode works without it
    llm_handler = None
    logger.info("Skipping LLM handler (VRAM conservation for 6GB card)")

    return dit_handler, llm_handler


def generate_one(dit_handler, llm_handler, variant, task_type, save_dir):
    """Generate a single track."""
    name = variant["name"]
    caption = variant["caption"]
    bpm = variant.get("bpm", 90)
    keyscale = variant.get("keyscale", "E major")

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating: {name} (task={task_type})")
    logger.info(f"{'='*60}")

    kwargs = dict(
        task_type=task_type,
        thinking=False,  # Disable CoT for VRAM/speed
        caption=caption,
        lyrics=CASEY_LYRICS,
        bpm=bpm,
        keyscale=keyscale,
        timesignature="4",
        vocal_language="en",
        duration=180.0,  # 3 minutes
        inference_steps=4,  # Turbo mode
        guidance_scale=3.0,
        seed=-1,
    )

    if task_type == "cover":
        kwargs["reference_audio"] = ORIGINAL_AUDIO
        kwargs["audio_cover_strength"] = variant.get("audio_cover_strength", 0.8)
        kwargs["cover_noise_strength"] = variant.get("cover_noise_strength", 0.3)

    params = GenerationParams(**kwargs)
    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
    )

    t0 = time.time()
    try:
        result = generate_music(
            dit_handler,
            llm_handler,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        elapsed = time.time() - t0

        if result.success:
            logger.info(f"SUCCESS: {name} — {elapsed:.1f}s")
            for audio in result.audios:
                logger.info(f"  -> {audio.get('path', '(in-memory)')}")
            return True
        else:
            logger.error(f"FAILED: {name} — {elapsed:.1f}s — {result.status_message}")
            return False
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"EXCEPTION: {name} — {elapsed:.1f}s — {e}")
        return False
    finally:
        gc.collect()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dit_handler, llm_handler = init_handlers()

    results = {}

    # Generate cover variants first (these use Casey's original as reference)
    for variant in COVER_VARIANTS:
        success = generate_one(dit_handler, llm_handler, variant, "cover", SAVE_DIR)
        results[variant["name"]] = "cover:" + ("OK" if success else "FAIL")
        gc.collect()

    # Generate text2music variants (from scratch with Casey's lyrics)
    for variant in TEXT2MUSIC_VARIANTS:
        success = generate_one(dit_handler, llm_handler, variant, "text2music", SAVE_DIR)
        results[variant["name"]] = "t2m:" + ("OK" if success else "FAIL")
        gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info("BATCH COMPLETE — Results Summary")
    logger.info(f"{'='*60}")
    for name, status in results.items():
        emoji = "✅" if "OK" in status else "❌"
        logger.info(f"  {emoji} {name}: {status}")

    logger.info(f"\nOutput directory: {SAVE_DIR}")


if __name__ == "__main__":
    main()
