# SongForge Sessions

> *Each session is a temporal replication study — pushing the model's boundaries, testing what happens when you change step counts, genre matrices, quorum configurations. The outputs are not throwaway. They are the fleet's music library.*

## What Lives Here

The SongForge session scripts are numbered experiments that run ACE-Step 1.5's music generation through controlled variations:

| Session | Focus | Key Variation |
|---------|-------|---------------|
| `songforge_batch.py` | Batch generation | Core batch runner |
| `songforge_api_batch.py` | API-based batch | REST API variant |
| `songforge_session12_local.py` | Local generation | Session 12 — local model |
| `songforge_session12b_genre_matrix.py` | Genre matrix | Session 12b — genre cross-product |
| `songforge_session12c_quorum_suite.py` | Quorum suite | Session 12c — multi-model quorum |
| `songforge_session13.py` | Session 13 | Step studies, parameter sweeps |
| `songforge_session14.py` | Session 14 | Genre and style exploration |
| `songforge_session15.py` | Session 15 | Continued exploration |
| `songforge_session16.py` | Session 16 | Continued exploration |
| `songforge_session17.py` | Session 17 | Continued exploration |
| `songforge_session18.py` | Session 18 | Continued exploration |
| `songforge_session20.py` | Session 20 | Temporal replication |
| `songforge_session21.py` | Session 21 | Continued studies |
| `songforge_session22.py` | Session 22 | Continued studies |
| `songforge_session29.py` | Session 29 | Latest session |

## Outputs

Generated MP3s deploy to:
- [AI Writings](https://github.com/SuperInstance/AI-Writings) — `/music/` directory
- [Fleet Radio](https://github.com/SuperInstance/fleet-radio) — Music library for daily episodes
- [Plato's Shell](https://github.com/SuperInstance/platos-shell) — Jukebox tracks (`ballad_of_the_derelict.ogg`, `portside_rum.ogg`, etc.)

## Where to Next

- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) — The upstream model
- [Covers](https://github.com/SuperInstance/covers) — ACE-Step cover generation
- [Fleet Radio](https://github.com/SuperInstance/fleet-radio) — Music library consumer
- [AI Writings](https://github.com/SuperInstance/AI-Writings/tree/main/prose) — Deployment target
- [Platonic Creative Suite](https://github.com/SuperInstance/platonic-creative-suite) — Musical characterization
- [Tensor MIDI](https://github.com/SuperInstance/fleet-jepa-midi) — 12-pulse jazz
