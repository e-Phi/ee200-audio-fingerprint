import os, json, time, collections
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa

app = Flask(__name__)
CORS(app)

# ── Load DB once at startup ───────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "fingerprint_db.json")

print(f"Loading database from {DB_PATH} …", flush=True)
t0 = time.time()
with open(DB_PATH, "r") as f:
    raw = json.load(f)

SONG_NAMES = {int(k): v for k, v in raw["song_names"].items()}
PARAMS     = raw["params"]
DB         = raw["db"]          # key: "f1,f2,dt" or "f1,dt"  →  [[sid, t1], …]

SR          = int(PARAMS.get("sample_rate", 22050))
N_FFT       = int(PARAMS.get("n_fft",       2048))
HOP_LENGTH  = int(PARAMS.get("hop_length",  512))
NEIGHBOR    = int(PARAMS.get("neighbor_size", 15))
AMP_FLOOR   = float(PARAMS.get("amp_floor_db", -60))
FAN_OUT     = int(PARAMS.get("fan_out",     15))
DT_MIN      = int(PARAMS.get("dt_min",      2))
DT_MAX      = int(PARAMS.get("dt_max",      200))
DF_MAX      = int(PARAMS.get("df_max",      200))

# Detect hash format from first key
_sample_key = next(iter(DB))
PAIR_MODE   = _sample_key.count(",") == 2      # True → "f1,f2,dt", False → "f1,dt"

print(f"DB loaded in {time.time()-t0:.1f}s  |  {len(DB):,} hashes  |  "
      f"{'pair' if PAIR_MODE else 'single'} mode  |  {len(SONG_NAMES)} songs", flush=True)


# ── DSP helpers ───────────────────────────────────────────────────────────────
def _spectrogram(y):
    S  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def _constellation(S_db):
    from scipy.ndimage import maximum_filter
    footprint = np.ones((NEIGHBOR, NEIGHBOR), dtype=bool)
    local_max = maximum_filter(S_db, footprint=footprint) == S_db
    above     = S_db > AMP_FLOOR
    peaks     = np.argwhere(local_max & above)   # [[freq_bin, time_frame], …]
    return peaks

def _hashes_pair(peaks):
    peaks = peaks[np.argsort(peaks[:, 1])]       # sort by time
    hashes = []
    for i, (f1, t1) in enumerate(peaks):
        for j in range(i+1, min(i+1+FAN_OUT, len(peaks))):
            f2, t2 = peaks[j]
            dt = int(t2 - t1)
            if dt < DT_MIN:  continue
            if dt > DT_MAX:  break
            if abs(int(f2)-int(f1)) > DF_MAX: continue
            hashes.append((int(f1), int(f2), dt, int(t1)))
    return hashes

def _hashes_single(peaks):
    hashes = []
    for f1, t1 in peaks:
        dt = 0
        hashes.append((int(f1), 0, dt, int(t1)))
    return hashes

def fingerprint_query(y):
    t_spec  = time.time()
    S_db    = _spectrogram(y)
    t_const = time.time()
    peaks   = _constellation(S_db)
    t_hash  = time.time()
    if PAIR_MODE:
        hashes  = _hashes_pair(peaks)
    else:
        hashes  = _hashes_single(peaks)
    t_lookup = time.time()

    offsets = collections.defaultdict(list)
    for entry in hashes:
        f1, f2, dt, t1 = entry
        key = f"{f1},{f2},{dt}" if PAIR_MODE else f"{f1},{dt}"
        if key in DB:
            for sid, ref_t1 in DB[key]:
                offset = ref_t1 - t1
                offsets[(int(sid), offset)].append(1)

    t_score = time.time()

    if not offsets:
        return None, {}, [], S_db.tolist(), peaks.tolist(), {
            "spectrogram_ms": round((t_const - t_spec)*1000),
            "constellation_ms": round((t_hash - t_const)*1000),
            "hashing_ms": round((t_lookup - t_hash)*1000),
            "lookup_ms": round((t_score - t_lookup)*1000),
            "scoring_ms": 0,
        }

    # Score: count agreeing hashes per (song, offset)
    song_scores = collections.defaultdict(int)
    best_offset_per_song = {}
    for (sid, offset), votes in offsets.items():
        c = len(votes)
        if c > song_scores[sid]:
            song_scores[sid] = c
            best_offset_per_song[sid] = offset

    best_sid = max(song_scores, key=song_scores.get)
    best_off = best_offset_per_song[best_sid]

    # Offset histogram for the winner
    hist = collections.Counter()
    for (sid, offset), votes in offsets.items():
        if sid == best_sid:
            hist[offset] += len(votes)

    t_end = time.time()

    timings = {
        "spectrogram_ms":  round((t_const  - t_spec)  * 1000),
        "constellation_ms":round((t_hash   - t_const) * 1000),
        "hashing_ms":      round((t_lookup - t_hash)  * 1000),
        "lookup_ms":       round((t_score  - t_lookup)* 1000),
        "scoring_ms":      round((t_end    - t_score) * 1000),
    }

    scores_out = {SONG_NAMES.get(sid, str(sid)): int(v)
                  for sid, v in sorted(song_scores.items(),
                                       key=lambda x: -x[1])[:8]}

    hist_out = {str(k): v for k, v in
                sorted(hist.items(), key=lambda x: -x[1])[:200]}

    return SONG_NAMES.get(best_sid, str(best_sid)), scores_out, peaks.tolist(), S_db.tolist(), peaks.tolist(), timings


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "songs": len(SONG_NAMES), "hashes": len(DB)})

@app.route("/library", methods=["GET"])
def library():
    return jsonify({"songs": SONG_NAMES})

@app.route("/identify", methods=["POST"])
def identify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    tmp  = f"/tmp/query_{int(time.time()*1000)}.audio"
    file.save(tmp)

    try:
        y, _ = librosa.load(tmp, sr=SR, mono=True, duration=30)
    except Exception as e:
        return jsonify({"error": f"Could not decode audio: {e}"}), 400
    finally:
        os.remove(tmp)

    match, scores, peaks, S_db, const_peaks, timings = fingerprint_query(y)

    if match is None:
        return jsonify({"match": None, "scores": {}, "timings": timings,
                        "peaks_count": 0})

    # Downsample spectrogram for transfer (every 4th col, every 2nd row)
    S_small = np.array(S_db)[::2, ::4].tolist()

    return jsonify({
        "match":       match,
        "scores":      scores,
        "timings":     timings,
        "peaks_count": len(peaks),
        "spectrogram": S_small,
        "constellation": [[int(p[0]), int(p[1])] for p in peaks[:2000]],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
