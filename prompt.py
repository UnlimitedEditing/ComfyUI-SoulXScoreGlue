"""Reference-voice metadata from raw audio, with no speech-recognition models.

SoulX-Singer clones a voice from a short sung reference (3-10 s) and needs that clip's metadata too (words, notes, pitch
curve). This builds it from the audio alone: pitch tracking -> notes -> one syllable per note (default "la"). Sing the
reference clip as a vowel line ("la la la", "ah", "ooh") for best results; it must be a clean, unaccompanied voice.
"""
import re

import numpy as np

FPS = 50.0          # pitch frames per second (hop 480 at 24 kHz)
SR = 24000
HOP = 480


def track_f0(y, sr=SR):
    import librosa
    f0, _vf, _vp = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr,
                                frame_length=2048, hop_length=HOP)
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float64)


def _runs(voiced):
    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))
    return runs


def segment_notes(f0, min_note_s=0.10, split_semitones=0.9, min_gap_s=0.08):
    """Voiced runs -> notes. A run is split where the smoothed pitch moves by more than `split_semitones`."""
    voiced = f0 > 0
    # close tiny unvoiced gaps so a consonant or tracker dropout does not cut a note
    gap = int(min_gap_s * FPS)
    v = voiced.copy()
    i = 0
    while i < len(v):
        if not v[i]:
            j = i
            while j < len(v) and not v[j]:
                j += 1
            if 0 < i and j < len(v) and (j - i) <= gap:
                v[i:j] = True
            i = j
        else:
            i += 1
    midi = np.where(f0 > 0, 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0)
    notes = []
    for a, b in _runs(v):
        cur_start = a
        ref = None
        for k in range(a, b):
            if midi[k] <= 0:
                continue
            if ref is None:
                ref = midi[k]
                continue
            if abs(midi[k] - ref) > split_semitones:
                # only split if the new pitch persists (not a one-frame glitch)
                look = midi[k:min(k + 4, b)]
                look = look[look > 0]
                if len(look) >= 3 and np.all(np.abs(look - midi[k]) < 0.6):
                    notes.append((cur_start, k))
                    cur_start, ref = k, midi[k]
            else:
                ref = 0.85 * ref + 0.15 * midi[k]
        notes.append((cur_start, b))
    out = []
    for a, b in notes:
        if (b - a) / FPS < min_note_s:
            continue
        seg = midi[a:b]
        seg = seg[seg > 0]
        if len(seg) == 0:
            continue
        out.append((a, b, int(round(float(np.median(seg))))))
    return out


def _word_phonemes(word):
    from g2p_en import G2p
    global _G
    try:
        _G
    except NameError:
        _G = G2p()
    ph = [p for p in _G(word) if re.match(r"^[A-Z]+[0-2]?$", p)]
    return "en_" + "-".join(ph) if ph else "en_L-AA1"


def build_prompt_metadata(y, sr=SR, lyrics="la", max_seconds=10.0, language="English"):
    """Returns (metadata dict, trimmed waveform) for a clean sung reference. y: mono float array at 24 kHz."""
    if len(y) > max_seconds * sr:
        # keep the most voiced contiguous max_seconds window
        f_all = track_f0(y, sr)
        win = int(max_seconds * FPS)
        voiced = (f_all > 0).astype(float)
        csum = np.concatenate([[0], np.cumsum(voiced)])
        best, best_i = -1, 0
        for i in range(0, max(1, len(voiced) - win), 5):
            sc = csum[min(i + win, len(voiced))] - csum[i]
            if sc > best:
                best, best_i = sc, i
        a = int(best_i / FPS * sr)
        y = y[a: a + int(max_seconds * sr)]
    f0 = track_f0(y, sr)
    notes = segment_notes(f0)
    if notes:
        # pyin sometimes jumps an octave on a note: pull outliers back toward the clip's median pitch (note AND curve)
        med = float(np.median([p for _a, _b, p in notes]))
        fixed = []
        for a, b, p in notes:
            shift = 0
            while p + shift > med + 9:
                shift -= 12
            while p + shift < med - 9:
                shift += 12
            if shift:
                f0[a:b] = f0[a:b] * (2.0 ** (shift / 12.0))
            fixed.append((a, b, p + shift))
        notes = fixed
    if not notes:
        raise ValueError("no sung notes found in the reference clip: use a clean, unaccompanied singing clip")
    words = [w for w in re.split(r"\s+", lyrics.strip()) if w] or ["la"]
    seq, last_end = [], 0
    pad = 0.0
    items = []  # (text, phoneme, pitch, type, dur)
    # leading rest
    first_a = notes[0][0]
    items.append(("<SP>", "<SP>", 0, 1, max(first_a / FPS, 0.05)))
    wi = 0
    last = (words[0], _word_phonemes(words[0].lower()))
    for idx, (a, b, p) in enumerate(notes):
        if idx > 0:
            gap = (a - notes[idx - 1][1]) / FPS
            if gap > 0.12:
                items.append(("<SP>", "<SP>", 0, 1, gap))
            elif gap > 0:
                # absorb a short gap into the previous note
                t, ph, pp, ty, d = items[-1]
                items[-1] = (t, ph, pp, ty, d + gap)
        dur = (b - a) / FPS
        if len(words) == 1:
            w = words[0]                         # a vowel line ("la la la"): the same syllable on every note
            items.append((w, _word_phonemes(w.lower()), p, 2, dur))
            last = (w, _word_phonemes(w.lower()))
        elif wi < len(words):
            w = words[wi]
            wi += 1
            last = (w, _word_phonemes(w.lower()))
            items.append((last[0], last[1], p, 2, dur))
        else:
            items.append((last[0], last[1], p, 3, dur))   # more notes than words: hold the last syllable
    tail = (len(f0) - notes[-1][1]) / FPS
    items.append(("<SP>", "<SP>", 0, 1, max(tail, 0.05)))
    total = sum(d for *_x, d in items)
    nfr = max(1, int(round(total * FPS)))
    f0_out = []
    acc = 0.0
    note_iter = iter(notes)
    # pitch curve: the tracked f0 itself (real vibrato / glides), zeros on rests
    f0_curve = f0[:nfr]
    f0_curve = np.concatenate([f0_curve, np.zeros(max(0, nfr - len(f0_curve)))])
    meta = {
        "index": f"vocal_0_{int(total * 1000)}",
        "language": language,
        "time": [0, int(round(total * 1000))],
        "duration": " ".join(f"{d:.2f}" for *_x, d in items),
        "text": " ".join(t for t, *_r in items),
        "phoneme": " ".join(ph for _t, ph, *_r in items),
        "note_pitch": " ".join(str(p) for _t, _ph, p, *_r in items),
        "note_type": " ".join(str(ty) for _t, _ph, _p, ty, _d in items),
        "f0": " ".join(f"{x:.1f}" for x in f0_curve),
    }
    return meta, y[: int(total * sr)]
