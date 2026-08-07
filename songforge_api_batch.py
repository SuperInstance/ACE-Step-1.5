#!/usr/bin/env python3
"""SongForge API Batch — generate covers via ACE-Step REST API.

Uses the already-running ACE-Step API server on port 8001.
Sends requests sequentially, saves outputs to experiments_v4/.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import shutil
import glob

API = "http://127.0.0.1:8001"
LYRICS = open("/home/eileen/projects/covers/casey_lyrics.txt").read().strip()
SAVE_DIR = "/home/eileen/projects/covers/experiments_v4"
CACHE_DIR = "/home/eileen/projects/ACE-Step-1.5/.cache/acestep/tmp/api_audio"

os.makedirs(SAVE_DIR, exist_ok=True)

# List existing cached audio files before we start
existing_cache = set(glob.glob(os.path.join(CACHE_DIR, "*")))

VARIANTS = [
    {
        "name": "v6_3am_kitchen",
        "prompt": (
            "Lo-fi bedroom folk recorded on a four-track cassette. "
            "The sound is intimate to the point of discomfort. "
            "Double-tracked vocals are slightly out of phase, creating a ghostly unison. "
            "A distant piano appears in the second verse, barely audible. "
            "The voice is a breathy male tenor, half-whispered. "
            "No drums. No bass. Just the skeleton of the song exposed. "
            "BPM: 72, Key: E major, Duration: 60 seconds."
        ),
    },
    {
        "name": "v6_gospel_hymn",
        "prompt": (
            "Gospel-inflected folk rock that builds from a single voice and acoustic guitar to a full choir. "
            "The opening verse is solo - just a clear male tenor with a slight rasp. "
            "By the first chorus, hand claps and a stomp board join. "
            "The second verse adds an upright bass. "
            "The final chorus explodes with a full gospel choir singing harmonies, a Hammond organ swelling. "
            "BPM: 100, Key: E major, Duration: 60 seconds."
        ),
    },
    {
        "name": "v6_celtic_ballad",
        "prompt": (
            "Traditional Celtic folk ballad adapted to contemporary indie production. "
            "Dropped-D acoustic guitar fingerpicking in E major, with uilleann pipes threading through. "
            "A bodhran frame drum provides a heartbeat rhythm on the choruses. "
            "The male voice is clear and unaffected, with a slight Irish inflection. "
            "Fiddle enters on the final chorus, playing counter-melodies. "
            "BPM: 80, Key: E major, Duration: 60 seconds."
        ),
    },
    {
        "name": "v6_blues_crossroads",
        "prompt": (
            "Delta blues meets indie folk. Open-E tuned acoustic guitar with a slide. "
            "The male voice is a gritty baritone that can break into falsetto on emotional peaks. "
            "A foot stomp on a wooden floor provides the only percussion. "
            "Harmonica accents punctuate the choruses, played through a cupped mic. "
            "The production is bone-dry. No reverb. Raw sound. "
            "BPM: 90, Key: E major, Duration: 60 seconds."
        ),
    },
    {
        "name": "v6_ambient_dreamscape",
        "prompt": (
            "Ambient folk soundscapes inspired by Bon Iver filtered through Sigur Ros. "
            "Acoustic guitar processed through reverse reverbs and granular delays. "
            "The male vocal is primarily in falsetto, treated with subtle auto-tune. "
            "Synth pads swell underneath. The choruses layer five or six vocal tracks. "
            "BPM: 60, Key: E major, Duration: 60 seconds."
        ),
    },
    {
        "name": "v6_chamber_meditation",
        "prompt": (
            "Chamber folk with the intimacy of Nick Drake and the orchestral ambition of Sufjan Stevens. "
            "A classical nylon-string guitar fingerpicks in E major. "
            "A cello holds a drone underneath. The male voice is whispered, close-miked. "
            "Piano notes fall like raindrops in the second verse. "
            "The chorus introduces a string quartet playing slow chords. No drums. "
            "BPM: 65, Key: E major, Duration: 60 seconds."
        ),
    },
]


def generate_one(variant):
    """Send a generation request to the ACE-Step API."""
    name = variant["name"]
    prompt = variant["prompt"]

    # Build the request like a chat completion
    req_body = {
        "model": "acestep/acestep-v15-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a music generation AI. Generate high-quality music based on the user description. Always include vocals with the provided lyrics.",
            },
            {
                "role": "user",
                "content": f"Generate music: {prompt}\n\nLyrics:\n{LYRICS}",
            },
        ],
        "stream": False,
    }

    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"[{name}] Sending request...", flush=True)
    t0 = time.time()

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            print(f"[{name}] Response received in {elapsed:.1f}s", flush=True)

            # Parse response
            try:
                result = json.loads(body)
                # Check for audio in response
                choices = result.get("choices", [])
                for choice in choices:
                    msg = choice.get("message", {})
                    audio = msg.get("audio", {})
                    if audio:
                        print(f"[{name}] Audio in response: {list(audio.keys())}", flush=True)

                # Print a summary
                print(f"[{name}] Response keys: {list(result.keys())}", flush=True)
                if "audio" in result:
                    print(f"[{name}] Top-level audio key found!", flush=True)
                if "data" in result:
                    print(f"[{name}] Data: {str(result['data'])[:200]}", flush=True)
            except json.JSONDecodeError:
                print(f"[{name}] Non-JSON response: {body[:500]}", flush=True)

            return True, result

    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        body = e.read().decode("utf-8", errors="replace")
        print(f"[{name}] HTTP {e.code} after {elapsed:.1f}s: {body[:300]}", flush=True)
        return False, {"error": body}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{name}] Error after {elapsed:.1f}s: {e}", flush=True)
        return False, {"error": str(e)}


def find_new_cache():
    """Find new audio files in the cache directory."""
    current_cache = glob.glob(os.path.join(CACHE_DIR, "*"))
    new_files = [f for f in current_cache if f not in existing_cache]
    return new_files


def main():
    results = {}

    for i, variant in enumerate(VARIANTS):
        name = variant["name"]
        print(f"\n{'='*60}", flush=True)
        print(f"  VARIANT {i+1}/{len(VARIANTS)}: {name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Update existing cache snapshot
        global existing_cache
        existing_cache = set(glob.glob(os.path.join(CACHE_DIR, "*")))

        success, response = generate_one(variant)

        if success:
            # Look for newly cached audio files
            new_files = find_new_cache()
            if new_files:
                for nf in new_files:
                    dest = os.path.join(SAVE_DIR, f"{name}.mp3")
                    shutil.copy2(nf, dest)
                    size = os.path.getsize(dest)
                    print(f"[{name}] Saved: {dest} ({size/1024:.1f} KB)", flush=True)
                    results[name] = f"OK ({size/1024:.0f}KB)"
            else:
                # Check if audio was inline in the response
                print(f"[{name}] No new cache file found. Checking response...", flush=True)
                results[name] = "OK (no file)"
        else:
            results[name] = f"FAIL: {str(response.get('error', ''))[:50]}"

        # Small delay between requests
        time.sleep(2)

    print(f"\n{'='*60}", flush=True)
    print("BATCH COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for name, status in results.items():
        print(f"  {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
