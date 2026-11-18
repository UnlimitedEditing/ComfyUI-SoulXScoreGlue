"""Compact score text -> SoulX-Singer metadata (the JSON the Advanced synthesizer reads).

Score text: whitespace-separated notes, each  word:midi:beats
    neon:62:1  lights:64:1  are:66:0.5  home:67:2  _:0:1  -:69:1
  word   an English word (phonemes come from g2p_en)
  _      a rest (midi ignored)
  -      hold: repeat the previous word on a new note (a held / melisma note)
  midi   MIDI note number (60 = middle C)
  beats  length in beats (quarter notes); seconds = beats * 60 / bpm
No '/' anywhere, so it survives Graydient's option parser.
"""
import json
import math
import os
import re

FPS = 50.0  # SoulX-Singer pitch frames per second (hop 480 at 24 kHz)

_g2p = None


def _phonemes(word):
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    ph = [p for p in _g2p(word) if re.match(r"^[A-Z]+[0-2]?$", p)]
    return "en_" + "-".join(ph) if ph else None


def parse_score(text):
    notes = []
    prev = None
    for tok in text.replace("\n", " ").replace(",", " ").split():
        parts = tok.split(":")
        if len(parts) < 3:
            raise ValueError(f"bad note '{tok}': expected word:midi:beats")
        word, midi, beats = parts[0], int(float(parts[1])), float(parts[2])
        if word == "_":
            notes.append({"text": "<SP>", "phoneme": "<SP>", "pitch": 0, "type": 1, "beats": beats})
        elif word == "-":
            if prev is None:
                raise ValueError("'-' (hold) cannot start a score")
            notes.append({"text": prev["text"], "phoneme": prev["phoneme"], "pitch": midi, "type": 3, "beats": beats})
        else:
            ph = _phonemes(word.lower())
            if ph is None:
                raise ValueError(f"no phonemes for '{word}'")
            n = {"text": word, "phoneme": ph, "pitch": midi, "type": 2, "beats": beats}
            notes.append(n)
            prev = n
    return notes


def build_segments(text, bpm, start_seconds=0.0, max_segment_seconds=8.0, language="English"):
    """Split the notes into segments (at rests where possible) and build one metadata dict per segment.
    Every note keeps its absolute time on the song timeline, so the synthesized audio lands exactly where the score says.
    A rest at a split point is shared by both segments (silence over silence), so no timing shifts."""
    spb = 60.0 / bpm
    notes = parse_score(text)
    t = float(start_seconds)
    for n in notes:
        n["dur"] = n["beats"] * spb
        n["t0"] = t
        t += n["dur"]
    pad = 0.12
    first = notes[0]
    if first["type"] != 1:
        p0 = min(pad, max(first["t0"], 0.0))
        notes.insert(0, {"text": "<SP>", "phoneme": "<SP>", "pitch": 0, "type": 1, "beats": 0.0, "dur": p0, "t0": first["t0"] - p0})
    last = notes[-1]
    if last["type"] != 1:
        notes.append({"text": "<SP>", "phoneme": "<SP>", "pitch": 0, "type": 1, "beats": 0.0, "dur": pad, "t0": last["t0"] + last["dur"]})
    segs, cur, cur_dur = [], [], 0.0
    for i, n in enumerate(notes):
        cur.append(n)
        cur_dur += n["dur"]
        is_last = i == len(notes) - 1
        if is_last:
            break
        if n["type"] == 1 and cur_dur >= max_segment_seconds * 0.5:
            segs.append(cur)
            cur, cur_dur = [n], n["dur"]          # share the rest: it opens the next segment too
        elif cur_dur >= max_segment_seconds:
            segs.append(cur)
            cur, cur_dur = [], 0.0
    if cur:
        segs.append(cur)
    out = []
    for seg in segs:
        total = sum(n["dur"] for n in seg)
        frames = max(1, int(round(total * FPS)))
        f0, acc = [], 0.0
        for n in seg:
            acc += n["dur"]
            upto = int(round(acc * FPS))
            hz = 0.0 if n["pitch"] <= 0 else 440.0 * 2.0 ** ((n["pitch"] - 69) / 12.0)
            f0.extend([hz] * max(0, upto - len(f0)))
        f0 = (f0 + [0.0] * frames)[:frames]
        start_ms = int(round(seg[0]["t0"] * 1000))
        end_ms = start_ms + int(round(total * 1000))
        out.append({
            "index": f"vocal_{start_ms}_{end_ms}",
            "language": language,
            "time": [start_ms, end_ms],
            "duration": " ".join(f"{n['dur']:.2f}" for n in seg),
            "text": " ".join(n["text"] for n in seg),
            "phoneme": " ".join(n["phoneme"] for n in seg),
            "note_pitch": " ".join(str(n["pitch"]) for n in seg),
            "note_type": " ".join(str(n["type"]) for n in seg),
            "f0": " ".join(f"{x:.1f}" for x in f0),
        })
    return out


def write_metadata(text, bpm, start_seconds, directory, name="target"):
    segs = build_segments(text, bpm, start_seconds)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segs, f, ensure_ascii=False)
    return path, segs[-1]["time"][1] / 1000.0
