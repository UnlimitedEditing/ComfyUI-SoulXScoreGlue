"""ComfyUI-SoulXScoreGlue: feed a plain-text score (lyrics + notes + timing) to SoulX-Singer's Advanced synthesizer.

Pairs with Saganaki22/ComfyUI-SoulX-Singer (SoulX-Singer Model Loader + SoulX-Singer Advanced). This pack writes the
two metadata JSON files that node reads, so a song line can be sung from text, with exact timing on the song timeline.
"""
import os

try:
    import folder_paths
except ImportError:  # unit tests outside ComfyUI
    folder_paths = None

from . import score as _score

_HERE = os.path.dirname(os.path.abspath(__file__))


def _temp_dir():
    base = folder_paths.get_temp_directory() if folder_paths else os.path.join(_HERE, "_tmp")
    d = os.path.join(base, "soulx_score")
    os.makedirs(d, exist_ok=True)
    return d


def _ensure_nltk():
    # the SoulX-Singer wrapper bundles nltk_data next to itself; use it when present so g2p_en never downloads at run time
    if os.environ.get("NLTK_DATA"):
        return
    base = os.path.dirname(_HERE)
    for name in os.listdir(base) if os.path.isdir(base) else []:
        cand = os.path.join(base, name, "nltk_data")
        if "soulx" in name.lower() and os.path.isdir(cand):
            os.environ["NLTK_DATA"] = cand
            return


class SoulXScoreTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "score": ("STRING", {"multiline": True, "default":
                      "_:0:1 neon:62:1 lights:64:1 are:66:0.5 call:67:0.5 ing:66:1 me:62:1 home:69:2 _:0:1"}),
            "bpm": ("FLOAT", {"default": 124.0, "min": 40.0, "max": 240.0, "step": 0.01}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            "example_target": ("STRING", {"default": "none"}),
        }}

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("target_metadata_path", "end_seconds", "info")
    FUNCTION = "run"
    CATEGORY = "SoulX-Singer"

    def run(self, score, bpm, start_seconds, example_target="none"):
        if example_target and example_target.strip().lower() != "none":
            # a SoulX-Singer shipped example (real singing metadata) for A/B quality checks of the model itself
            p = os.path.join(_HERE, "voices", f"{example_target.strip()}.json")
            print(f"[SoulXScoreTarget] using shipped example target {p}")
            return (p, 0.0, f"example target {example_target}")
        _ensure_nltk()
        path, end = _score.write_metadata(score, bpm, start_seconds, _temp_dir(), "target")
        info = f"segments written to {path}; line ends at {end:.2f}s"
        print(f"[SoulXScoreTarget] {info}")
        return (path, float(end), info)


class SoulXPromptVoice:
    """The reference voice. With no audio it is the built-in example voice; with a clean sung clip it is a clone of that
    singer (metadata is built from the audio itself: pitch tracking -> notes -> one syllable per note)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"voice": (["en_prompt"], {"default": "en_prompt"}),
                             "lyrics": ("STRING", {"default": "la"})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("prompt_audio", "prompt_metadata_path", "info")
    FUNCTION = "run"
    CATEGORY = "SoulX-Singer"

    def run(self, voice, lyrics="la", audio=None):
        import numpy as np
        import soundfile as sf
        import torch
        if audio is not None and audio.get("waveform") is not None and audio["waveform"].numel() > 0:
            import json
            import librosa
            from . import prompt as _prompt
            _ensure_nltk()
            w = audio["waveform"][0].detach().cpu().numpy().astype(np.float32)   # [C, T]
            y = w.mean(axis=0)
            sr = int(audio["sample_rate"])
            if sr != _prompt.SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=_prompt.SR)
            peak = float(np.abs(y).max()) or 1.0
            y = y / peak * 0.9
            meta, yy = _prompt.build_prompt_metadata(y, _prompt.SR, lyrics or "la")
            path = os.path.join(_temp_dir(), "prompt_from_audio.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([meta], f, ensure_ascii=False)
            out = {"waveform": torch.from_numpy(np.ascontiguousarray(yy.astype(np.float32)))[None, None], "sample_rate": _prompt.SR}
            info = f"cloned reference: {len(yy) / _prompt.SR:.1f}s, notes {meta['note_pitch']}"
            print(f"[SoulXPromptVoice] {info}")
            return (out, path, info)
        wav = os.path.join(_HERE, "voices", f"{voice}.wav")
        meta = os.path.join(_HERE, "voices", f"{voice}.json")
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        out = {"waveform": torch.from_numpy(np.ascontiguousarray(y.T))[None], "sample_rate": int(sr)}
        return (out, meta, f"built-in voice {voice}")


NODE_CLASS_MAPPINGS = {"SoulX Score Target": SoulXScoreTarget, "SoulX Prompt Voice": SoulXPromptVoice}
NODE_DISPLAY_NAME_MAPPINGS = {"SoulX Score Target": "SoulX - Score to Target Metadata",
                              "SoulX Prompt Voice": "SoulX - Built-in Prompt Voice"}
