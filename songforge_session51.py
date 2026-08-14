#!/usr/bin/env python3
"""SongForge Session 51 — The DeepSeek Prompt Protocol / The 51st Tail

Wednesday 10:46 PM AKST, August 12 2026. MMX quota exhausted (resets Aug 17).
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

EXPERIMENT A: DeepSeek-Style Structured Prompts
Using ultra-detailed production prompts inspired by DeepSeek's style —
specifying microphone placement, room acoustics, signal chain, and
production technique. The question: does ACE-Step respond to engineering-
level detail the way it responds to genre tags?

EXPERIMENT B: Genre Mutation Matrix 3
New impossible hybrids never tested:
1. "Mathematical sea shanty" — compound meter + accordion + ocean
2. "Baroque vaporwave" — harpsichord + slowed tape + reverb drench
3. "Industrial lullaby" — metallic percussion + soft vocals + nursery melody

EXPERIMENT C: Same Lyrics, Three Models
The fiftieth-session lyrics (Phi3 version) rendered through three
different ACE-Step checkpoints: turbo, 1.7B, and 0.6B.
This is the first direct checkpoint comparison.

EXPERIMENT D: The Temperature Prompt Chain
Llama3.2 at three temperatures generates music captions.
Same concept, different thermal regimes.
Each caption becomes an instrumental ACE-Step track.
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

# From Session 50 — the fiftieth session lyrics (Phi3 version, most expansive)
FIFTIETH_LYRICS = """[Verse]
In the first room, a tuning fork struck flint
and the spark became a scale
In the second room, a metronome forgot its tempo
and found a new one in the silence between beats
In the third room, a microphone heard its own ghost
and the ghost was singing

[Chorus]
Fifty rooms, fifty resonances
The corridor is a chord
Each door is a harmonic
The hallway is the fundamental frequency

[Verse]
In the twenty-fifth room, the spectrum inverted
and silence became the loudest sound
In the thirty-sixth room, materials began to sing
copper, glass, steel, ice — each with its own voice
In the fiftieth room, the census taker arrived
and found that every room had already been counted
by its own echo

[Bridge]
The frequency is not a number
It is a relationship between a room and its contents
Between a wave and its boundary
Between a sound and the silence it modifies

[Outro]
Fifty rooms. Fifty doors. Fifty resonances.
The corridor sings.
The census is complete.
The listener is at the door.
The listener is the fifty-first room.
"""


# ═══════════ DEEPSEEK-STYLE PROMPTS ═══════════
# These are manually crafted in the DeepSeek structured style:
# specifying recording technique, room, signal chain, production

DEEPSEEK_PROMPTS = [
    {
        "name": "close-miked-cello",
        "prompt": "Close-miked cello in a small concrete practice room, microphone placed 30cm from the bridge, dry signal with subtle room reflections at 1.2s reverb tail, slow sustained bowing in the lower register, occasional harmonics produced by lightly touching the string at the nodal point, the player breathes audibly between phrases, a faint 60Hz electrical hum from the amplifier underneath, recorded on analog tape with slight wow and flutter, stereo image is narrow and intimate",
        "lyrics": FIFTIETH_LYRICS,
    },
    {
        "name": "prepared-piano-stairwell",
        "prompt": "Prepared piano in a concrete stairwell, objects placed on the strings: paper between A4 and A5, a small glass tumbler resting on the C3 string, metal screws on the F5 and G5 strings, each note decays differently depending on the object, the stairwell adds a 3.5 second reverb with distinct early reflections from each landing, recorded with a stereo pair of small diaphragm condensers in ORTF configuration on the third floor landing, the pianist plays slowly and deliberately, letting each note decay fully before the next",
        "lyrics": FIFTIETH_LYRICS,
    },
    {
        "name": "contact-mic-bridge",
        "prompt": "Contact microphone attached to the steel suspension cables of a bridge, the cables vibrate at their natural frequency in the wind, the sound is a deep metallic drone with upper harmonics that shift as wind speed changes, a passing truck excites a momentary 17Hz rumble that fades over 8 seconds, recorded at 3am with no traffic for the middle section, only wind and the creaking of expansion joints, the stereo field is created by two contact mics on different cables tuned a perfect fifth apart",
        "lyrics": FIFTIETH_LYRICS,
        "instrumental": True,
    },
]


# ═══════════ GENRE MUTATION MATRIX ═══════════

GENRE_MUTATIONS = [
    {
        "name": "mathematical-sea-shanty",
        "prompt": "Mathematical sea shanty in 7/8 time, accordion and concertina playing interlocking polyrhythmic patterns, a deep male voice singing a work song with the cadence of hauling rope but the precision of a fugue, bodhran frame drum accentuating the unusual meter, fiddle playing pizzicato counterpoint in the second voice, the harmonic language shifts between Dorian and Mixolydian modes every four bars, the production is warm and wood-paneled like a ship's interior, tempo 108 BPM",
        "lyrics": """[Verse]
Heave away on the one-two-three
The seventh beat is the wave that sets you free
The ocean doesn't count in fours
It counts in sevens and then it pours

[Chorus]
Roll, roll, the mathematical sea
Every wave is a proof by induction
The ship is a lemma
The harbor is the theorem
And the journey is the proof

[Verse]
The navigator draws fractals on the chart
Each coastline branches into smaller coastlines
The compass needle oscillates between true and magnetic
Between the ideal and the real
Between where we are and where we meant to be

[Outro]
Heave away on the one-two-three-four-five-six-seven
The sea is a counting problem with no solution
The ship solves it anyway
The ship is the solution
""",
    },
    {
        "name": "baroque-vaporwave",
        "prompt": "Baroque vaporwave, harpsichord and viola da gamba playing a Handel-inspired chord progression but slowed to half speed with heavy tape modulation, the reverb tail is impossibly long — cathedral acoustics stretched beyond physical possibility, the harpsichord's pluck dissolves into a warm pad, pitched down vocals drift through the mix like ghosts of castrati, the entire production has the warm hiss and compression of a VHS tape, the key modulates up a major third in the final section creating a sense of ascending through layers of dream",
        "lyrics": """[Verse]
The garden is a algorithm
Each hedge a line of code
The fountain computes prime numbers
In the language of falling water

[Chorus]
In the court of the Sun King
The mirrors reflect infinitely
Each reflection slightly degraded
Each copy an homage
Each copy a betrayal

[Verse]
The harpsichord plays a figured bass
But the figures are hexadecimal
The basso continuo is a Turing machine
Reading the score as if it were a tape
The notes are both music and instruction

[Outro]
The garden grows slower each century
The roses bloom in geological time
The palace dissolves into pixels
The pixels dissolve into sound
""",
    },
    {
        "name": "industrial-lullaby",
        "prompt": "Industrial lullaby, metallic percussion created by striking steel pipes and metal sheets with varying force, the rhythm mimics a heartbeat that gradually slows, a soft female voice sings a simple nursery-rhyme melody in Dorian mode, the vocal is recorded with a vintage ribbon microphone for warmth, underneath everything is a sub-bass drone at 55Hz that modulates in amplitude like breathing, the metallic sounds are processed through a spring reverb giving them an unsettling liquid quality, the overall effect should be simultaneously soothing and menacing, like being rocked to sleep in a factory",
        "lyrics": """[Verse]
Hush little one, the forge is cooling
The molten iron is becoming steel
The hammers have stopped their conversation
The anvil dreams of horseshoes

[Chorus]
Sleep now, the gears are lubricated
The assembly line has finished its song
The night shift watches over the foundry
And the foundry watches over you

[Verse]
The crane swings slowly overhead
Carrying the day's production
Each link in the chain was forged by hand
Each hand is now at rest
Each rest is a kind of music

[Outro]
Hush, the cooling tower hums
The turbine spins down
The last spark dies in the dust
The dust settles on the sleeping town
""",
    },
]


# ═══════════ TEMPERATURE PROMPT CHAIN ═══════════
# We'll generate these via Ollama first, then feed to ACE-Step

TEMP_CONCEPT = "The fifty-first room is not a room at all. It is the corridor between the other fifty rooms. The corridor has its own resonance — lower than any room, because it is longer. The corridor is the fundamental frequency that all fifty rooms are harmonics of."

TEMP_PROMPTS = {
    "t05": "Ambient drone in C major, sustained organ tones blending with field recording of distant hallway acoustics, 60 BPM, soft attack, infinite decay, the sound of air conditioning hum modulating with the room tone, pure and simple, minimal, contemplative",
    "t08": "Ambient electronic with irregular heartbeat rhythm, processed field recording of a long corridor reverberating, granular synthesis stretching a single piano note into a 4-minute texture, sub-bass at 55 Hz pulses like breathing, 73 BPM with constant tempo drift, the acoustic shadow of footsteps in a hallway",
    "t11": "Glitch ambient, the corridor sound is decomposed into spectral fragments and reassembled wrong, a piano note is stretched until it becomes noise then compressed back into a note then stretched again, ring modulation creates sidebands that sound like distant conversations, time stretches and compresses unpredictably, the heartbeat rhythm is subdivided into polyrhythms that never resolve, 97 BPM decelerating to 43 BPM",
}


def generate_with_acestep(handler, prompt, lyrics, duration=90, instrumental=False, seed=None, cuda=True):
    """Generate a single track with ACE-Step."""
    try:
        params = GenerationParams(
            audio_duration=duration,
            prompt=prompt,
            lyrics="" if instrumental else lyrics,
            infer_step=36,  # turbo uses fewer steps
            guidance_scale=7.5,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            actual_infer_steps=36,
            karras_sigmas=True,
            minimal_filter=True,
        )

        config = GenerationConfig(
            ckpoint_path=CHECKPOINT_DIR,
            cuda=cuda,
            device="cuda" if cuda else "cpu",
            dtype="bfloat16" if cuda else "float32",
            vae_offload=True,  # RTX 4050 needs this
            param_offload=True,
            lpips_offload=True,
            max_depth=45,
            resale_log_log_fn=lambda x: logger.info(x),
        )

        output_path = generate_music(
            params=params,
            config=config,
            handler=handler,
            save_path=SAVE_DIR,
            return_data=False,
            seed=seed,
        )
        return output_path
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        traceback.print_exc()
        return None


def run_session():
    """Run Session 51 experiments."""
    logger.info("=" * 60)
    logger.info("SongForge Session 51 — The DeepSeek Prompt Protocol")
    logger.info("=" * 60)

    # Initialize handler with turbo checkpoint
    logger.info("Loading ACE-Step turbo checkpoint...")
    handler = AceStepHandler(
        ckpoint_path=os.path.join(CHECKPOINT_DIR, "acestep-v15-turbo"),
        cuda=True,
        dtype="bfloat16",
        vae_offload=True,
        param_offload=True,
        lpips_offload=True,
    )
    logger.info("Handler loaded successfully.")

    results = []

    # ═══════════ EXPERIMENT A: DeepSeek-Style Prompts ═══════════
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT A: DeepSeek-Style Structured Prompts")
    logger.info("=" * 40)

    for i, item in enumerate(DEEPSEEK_PROMPTS):
        logger.info(f"\nTrack A{i+1}: {item['name']}")
        logger.info(f"  Prompt: {item['prompt'][:100]}...")
        start = time.time()
        path = generate_with_acestep(
            handler,
            prompt=item["prompt"],
            lyrics=item.get("lyrics", ""),
            duration=90,
            instrumental=item.get("instrumental", False),
            seed=51 + i,
        )
        elapsed = time.time() - start
        logger.info(f"  Generated in {elapsed:.1f}s → {path}")
        results.append({
            "exp": "A",
            "track": f"A{i+1}",
            "name": item["name"],
            "path": path,
            "time": elapsed,
            "prompt_type": "deepseek_structured",
        })
        # Cleanup between runs
        gc.collect()
        if hasattr(handler, 'cleanup_kv_cache'):
            handler.cleanup_kv_cache()


    # ═══════════ EXPERIMENT B: Genre Mutation Matrix 3 ═══════════
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT B: Genre Mutation Matrix 3")
    logger.info("=" * 40)

    for i, item in enumerate(GENRE_MUTATIONS):
        logger.info(f"\nTrack B{i+1}: {item['name']}")
        logger.info(f"  Prompt: {item['prompt'][:100]}...")
        start = time.time()
        path = generate_with_acestep(
            handler,
            prompt=item["prompt"],
            lyrics=item["lyrics"],
            duration=120,
            seed=51 + 10 + i,
        )
        elapsed = time.time() - start
        logger.info(f"  Generated in {elapsed:.1f}s → {path}")
        results.append({
            "exp": "B",
            "track": f"B{i+1}",
            "name": item["name"],
            "path": path,
            "time": elapsed,
            "prompt_type": "impossible_genre",
        })
        gc.collect()
        if hasattr(handler, 'cleanup_kv_cache'):
            handler.cleanup_kv_cache()


    # ═══════════ EXPERIMENT C: Temperature Prompt Chain ═══════════
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT C: Temperature Prompt Chain")
    logger.info("=" * 40)

    for temp_key, temp_prompt in TEMP_PROMPTS.items():
        logger.info(f"\nTrack C-{temp_key}: Temperature prompt")
        logger.info(f"  Prompt: {temp_prompt[:100]}...")
        start = time.time()
        path = generate_with_acestep(
            handler,
            prompt=temp_prompt,
            lyrics="",  # instrumental — clean spectral analysis
            duration=90,
            instrumental=True,
            seed=51 + 20,
        )
        elapsed = time.time() - start
        logger.info(f"  Generated in {elapsed:.1f}s → {path}")
        results.append({
            "exp": "C",
            "track": f"C-{temp_key}",
            "name": f"temperature-{temp_key}",
            "path": path,
            "time": elapsed,
            "prompt_type": "temperature_chain",
        })
        gc.collect()
        if hasattr(handler, 'cleanup_kv_cache'):
            handler.cleanup_kv_cache()

    # ═══════════ EXPERIMENT D: Cross-Checkpoint Comparison ═══════════
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT D: Cross-Checkpoint Comparison")
    logger.info("=" * 40)

    # We need to reload the handler with different checkpoints
    for ckpt_name, ckpt_path in [
        ("1.7B", os.path.join(CHECKPOINT_DIR, "acestep-5Hz-lm-1.7B")),
        ("0.6B", os.path.join(CHECKPOINT_DIR, "acestep-5Hz-lm-0.6B")),
    ]:
        logger.info(f"\nLoading checkpoint: {ckpt_name}")
        try:
            handler2 = AceStepHandler(
                ckpoint_path=ckpt_path,
                cuda=True,
                dtype="bfloat16",
                vae_offload=True,
                param_offload=True,
                lpips_offload=True,
            )

            prompt = "Ambient orchestral, sustained strings and soft piano, D minor, 70 BPM, cinematic, building slowly from nothing to a gentle peak and fading, the sound of a large empty hall"
            start = time.time()
            path = generate_with_acestep(
                handler2,
                prompt=prompt,
                lyrics=FIFTIETH_LYRICS,
                duration=90,
                seed=51,
            )
            elapsed = time.time() - start
            logger.info(f"  {ckpt_name} generated in {elapsed:.1f}s → {path}")
            results.append({
                "exp": "D",
                "track": f"D-{ckpt_name}",
                "name": f"checkpoint-{ckpt_name}",
                "path": path,
                "time": elapsed,
                "prompt_type": "cross_checkpoint",
            })
            del handler2
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Checkpoint {ckpt_name} failed: {e}")
            traceback.print_exc()

    # ═══════════ SAVE RESULTS ═══════════
    results_path = os.path.join(SAVE_DIR, "session51_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Session 51 complete. {len(results)} tracks generated.")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    run_session()
