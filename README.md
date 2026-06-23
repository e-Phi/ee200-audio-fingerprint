# EE200 Audio Fingerprinting — Zappatin-America

A browser-based audio identification system in the style of Shazam, built for **EE200: Signals, Systems & Networks (Q3B)**. The application identifies a song from a short audio clip by matching a sparse constellation of spectrogram peaks against a pre-indexed hash database — entirely client-side, with no backend server.

**Live app:** https://e-phi.github.io/ee200-audio-fingerprint/

## Overview

Given an audio clip, the app:

1. Decodes and resamples the audio to 11,025 Hz using the Web Audio API.
2. Computes a short-time Fourier transform (STFT) spectrogram (`N_FFT = 4096`, hop `= 512`).
3. Extracts local-maximum peaks from the spectrogram to form a sparse constellation.
4. Pairs nearby peaks into `(f1, f2, Δt)` hashes and looks them up in a pre-built fingerprint database.
5. Tallies votes by time offset; the song whose hashes align at a single dominant offset is reported as the match.

All signal-processing constants (sample rate, FFT size, hop length, peak-picking neighbourhood, hashing parameters) match the Python reference implementation used to build the database, so a clip fingerprinted in-browser produces hashes directly comparable to the indexed library.

## Features

The app is organised into three tabs:

| Tab | Description |
|---|---|
| **Library** | Browse the 50 pre-indexed songs, each shown with its hash count and a constellation thumbnail. |
| **Identify** | Upload a single query clip and view the recognised song alongside its spectrogram, constellation map, the corresponding window in the matched song, and the offset histogram that decided the match. |
| **Batch** | Upload multiple query clips at once; each is identified independently and the results can be downloaded as `results.csv` (`filename, prediction`). |

## Repository Structure

```
.
├── index.html              # Single-page application (UI, DSP, matching — no build step)
├── fingerprint_db.json      # Pre-computed hash database for the 50-song library
├── netlify.toml             # Alternate deployment configuration (Netlify)
└── .github/workflows/       # GitHub Actions workflow for GitHub Pages deployment
```

## Running Locally

No build tools or dependencies are required.

```bash
git clone https://github.com/e-Phi/ee200-audio-fingerprint.git
cd ee200-audio-fingerprint
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser. (A local server is required because `fingerprint_db.json` is loaded via `fetch`, which most browsers block on the `file://` protocol.)

## Background

This app is the deployment companion to a Jupyter notebook that develops and validates the underlying fingerprinting algorithm in Python (spectrogram analysis, constellation extraction, hash-based indexing, and robustness testing against noise and pitch shift). The notebook's hyperparameters and hashing scheme are reproduced exactly in this client-side JavaScript implementation so that the deployed database and the in-browser matcher remain consistent.

## License

Coursework submission for EE200 — Signals, Systems & Networks.
