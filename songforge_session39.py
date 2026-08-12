#!/usr/bin/env python3
"""SongForge Session 39 — The Physical Phenomena Experiment

Tuesday 11:00 AM AKST, August 11 2026. MMX daily quota at 0%.
ACE-Step 1.5 turbo on RTX 4050 (6GB VRAM, CPU VAE offload).

EXPERIMENT A: Physical Phenomena Prompt Chain
Four local LLMs describe music through PURELY PHYSICAL phenomena.
Each model's prompts become ACE-Step captions for instrumental tracks.

EXPERIMENT B: Cross-LLM Lyricist
Same physical captions, but lyrics from a DIFFERENT LLM.

EXPERIMENT C: Temperature Gradient
Best physical prompt with lyrics at 3 different LLM temperatures.

All tracks are 90 seconds, instrumental for Exp A (clean spectral analysis),
with vocals for Exp B & C.
"""

import json
import os
import sys
import time
import gc
import subprocess
import traceback
import glob

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

# ═══════════ LLM PROMPT GENERATION ═══════════

PHYSICAL_CONSTRAINT = """You are describing music. You must use ONLY physical phenomena — matter, energy, forces, motion, temperature, pressure, texture. NO emotion words. NO psychological states. NO mood words.

Instead of "sad," describe a heavy stone sinking in still water.
Instead of "joyful," describe a ball bouncing higher with each bounce.

Write exactly 3 different music descriptions, each 2-4 sentences. Each should evoke a different feeling through purely physical imagery. Format as a JSON array of 3 strings."""

LLMS = [
    {"name": "llama", "model": "llama3.2", "temp": 0.8},
    {"name": "phi3", "model": "phi3", "temp": 0.7},
    {"name": "qwen", "model": "qwen2.5:3b", "temp": 0.7},
    {"name": "granite", "model": "granite3.1-dense:2b", "temp": 0.7},
]

LYRIC_CONSTRAINT = """Write short song lyrics (2 verses + 1 chorus, under 400 characters total) inspired by this physical music description. Use only physical imagery — no emotion words. Describe physical events that imply feeling through matter, motion, and force.

Music description: {caption}"""

KEY_BPM = [
    ("D minor", 70),
    ("A minor", 65),
    ("C major", 80),
    ("E minor", 90),
    ("G major", 75),
    ("F major", 60),
    ("B minor", 100),
    ("A major", 85),
    ("D major", 70),
    ("E major", 95),
]

def generate_with_ollama(model, prompt, temperature=0.7):
    """Generate text with an Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=90,
            env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"}
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Ollama generation failed for {model}: {e}")
        return ""

def parse_prompts(text):
    """Extract up to 3 prompts from LLM output."""
    text = text.strip()
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            arr = json.loads(text[start:end])
            return [str(x).strip() for x in arr if len(str(x).strip()) > 30][:3]
    except:
        pass
    # Fallback: split by double newlines
    blocks = [b.strip() for b in text.split('\n\n') if len(b.strip()) > 50]
    return blocks[:3]

# ═══════════ BUILD TRACKS ═══════════

def build_experiment_a_tracks():
    """Physical phenomena prompts → instrumental ACE-Step tracks."""
    tracks = []
    key_idx = 0
    
    for llm in LLMS:
        logger.info(f"Generating physical prompts with {llm['name']} ({llm['model']})...")
        raw = generate_with_ollama(llm['model'], PHYSICAL_CONSTRAINT, llm['temp'])
        prompts = parse_prompts(raw)
        
        if not prompts:
            logger.warning(f"No prompts parsed from {llm['name']}")
            continue
        
        logger.info(f"  Got {len(prompts)} prompts from {llm['name']}")
        for i, p in enumerate(prompts):
            logger.info(f"    [{i+1}] {p[:80]}...")
        
        prompt_file = os.path.join(SAVE_DIR, f"s39-{llm['name']}-prompts.json")
        with open(prompt_file, 'w') as f:
            json.dump({"model": llm['name'], "raw": raw, "parsed": prompts}, f, indent=2)
        
        for i, prompt in enumerate(prompts):
            key, bpm = KEY_BPM[key_idx % len(KEY_BPM)]
            key_idx += 1
            tracks.append({
                "name": f"s39-phys-{llm['name']}-{i+1}",
                "caption": prompt[:500],
                "lyrics": "",
                "duration": 90,
                "key": key,
                "bpm": bpm,
                "prompt_source": llm['name'],
                "experiment": "A_physical",
            })
    
    return tracks


def build_experiment_b_tracks(exp_a_tracks):
    """Cross-LLM lyricist: each caption gets lyrics from a different LLM."""
    by_llm = {}
    for t in exp_a_tracks:
        src = t["prompt_source"]
        if src not in by_llm:
            by_llm[src] = t
    
    llm_order = [l["name"] for l in LLMS if l["name"] in by_llm]
    if len(llm_order) < 2:
        return []
    
    tracks = []
    key_idx = 5  # offset for variety
    
    for i, llm_name in enumerate(llm_order):
        caption_track = by_llm[llm_name]
        lyricist_name = llm_order[(i + 1) % len(llm_order)]
        lyricist_model = next(l["model"] for l in LLMS if l["name"] == lyricist_name)
        
        logger.info(f"Cross: caption from {llm_name}, lyrics from {lyricist_name}...")
        lyric_prompt = LYRIC_CONSTRAINT.format(caption=caption_track["caption"])
        lyrics = generate_with_ollama(lyricist_model, lyric_prompt, 0.8)
        
        if not lyrics or len(lyrics) < 50:
            logger.warning(f"  No lyrics generated, skipping cross track")
            continue
        
        key, bpm = KEY_BPM[key_idx % len(KEY_BPM)]
        key_idx += 1
        
        tracks.append({
            "name": f"s39-cross-{llm_name}-by-{lyricist_name}",
            "caption": caption_track["caption"][:500],
            "lyrics": lyrics[:1200],
            "duration": 90,
            "key": key,
            "bpm": bpm,
            "prompt_source": llm_name,
            "lyricist": lyricist_name,
            "experiment": "B_cross_lyricist",
        })
    
    return tracks


def build_experiment_c_tracks(exp_a_tracks):
    """Temperature gradient: best caption with lyrics at 3 temperatures."""
    # Prefer phi3, fall back to first available
    phi3_tracks = [t for t in exp_a_tracks if t["prompt_source"] == "phi3"]
    base = phi3_tracks[0] if phi3_tracks else (exp_a_tracks[0] if exp_a_tracks else None)
    if not base:
        return []
    
    base_caption = base["caption"]
    tracks = []
    
    for temp in [0.3, 0.7, 1.1]:
        logger.info(f"Temperature gradient: phi3 at temp {temp}...")
        lyric_prompt = LYRIC_CONSTRAINT.format(caption=base_caption)
        lyrics = generate_with_ollama("phi3", lyric_prompt, temp)
        
        if not lyrics or len(lyrics) < 50:
            continue
        
        tracks.append({
            "name": f"s39-temp-{str(temp).replace('.','')}",
            "caption": base_caption[:500],
            "lyrics": lyrics[:1200],
            "duration": 90,
            "key": "D minor",
            "bpm": 70,
            "prompt_source": "phi3",
            "lyricist": f"phi3-t{temp}",
            "llm_temperature": temp,
            "experiment": "C_temperature",
        })
    
    return tracks


# ═══════════ GENERATION ═══════════

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
        if isinstance(result_obj, dict) and "audio_path" in result_obj:
            output_path = result_obj["audio_path"]
        elif isinstance(result_obj, str):
            output_path = result_obj
        else:
            files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.mp3")),
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
                "size_mb": round(fsize/1024/1024, 2), "path": output_path}
        
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ✗ Failed in {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return {"name": name, "status": "error", "time": round(elapsed, 1), "error": str(e)}


def main():
    logger.info("=" * 60)
    logger.info("SongForge Session 39: The Physical Phenomena Experiment")
    logger.info("=" * 60)
    
    # Phase 1: Generate prompts with local LLMs
    logger.info("\n=== PHASE 1: LLM Prompt Generation ===")
    exp_a_tracks = build_experiment_a_tracks()
    logger.info(f"Experiment A: {len(exp_a_tracks)} physical phenomena tracks")
    
    exp_b_tracks = build_experiment_b_tracks(exp_a_tracks)
    logger.info(f"Experiment B: {len(exp_b_tracks)} cross-LLM lyricist tracks")
    
    exp_c_tracks = build_experiment_c_tracks(exp_a_tracks)
    logger.info(f"Experiment C: {len(exp_c_tracks)} temperature gradient tracks")
    
    all_tracks = exp_a_tracks + exp_b_tracks + exp_c_tracks
    logger.info(f"\nTotal tracks to generate: {len(all_tracks)}")
    
    if not all_tracks:
        logger.error("No tracks to generate! Check Ollama connectivity.")
        return
    
    # Phase 2: Load ACE-Step model
    logger.info("\n=== PHASE 2: Loading ACE-Step Model ===")
    
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
    
    # Phase 3: Generate tracks sequentially
    logger.info("\n=== PHASE 3: Music Generation ===")
    
    results = []
    for i, track in enumerate(all_tracks):
        logger.info(f"\n--- Track {i+1}/{len(all_tracks)}: {track['name']} ---")
        logger.info(f"  Experiment: {track.get('experiment', '?')}")
        result = generate_track(handler, track)
        results.append({**result, **{k: v for k, v in track.items() if k != "name"}})
        
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    # Phase 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("SESSION 39 SUMMARY")
    logger.info("=" * 60)
    
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "error")
    total_time = sum(r.get("time", 0) for r in results)
    
    logger.info(f"Generated: {ok} ok, {fail} failed")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        exp = r.get("experiment", "?")
        src = r.get("prompt_source", "?")
        size = r.get("size_mb", 0)
        logger.info(f"  {status} {r['name']} [{exp}/{src}] {size}MB ({r.get('time',0)}s)")
    
    results_file = os.path.join(SAVE_DIR, "s39-results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
