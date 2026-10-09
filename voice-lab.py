#!/usr/bin/env python3
"""A male voice from Heart's style vectors (John, 2026-10-09: "a true male voice that is based on
the Heart's voice style vectors; a baritone closer to the bass end than tenor").

A Kokoro voice pack is [510, 256]: row = phoneme count, 256 = 128 for the decoder's timbre
(ref_s[:, :128]) + 128 for the predictor's prosody, pitch and pacing (ref_s[:, 128:]).
Recipes tried here, each rendered to a WAV with its median pitch measured:

  shift   Heart plus a scaled male direction, the mean of the male packs minus the mean of the
          female packs, row by row. The direction moves timbre and prosody toward male while
          Heart's own offset (what makes her sound like her) is kept.
  halves  Heart's prosody half with a male voice's timbre half.
  f0      On top of either, the predicted pitch scaled (the generator's harmonic source follows).

    cd kokoro-coreml-export && uv run python ../voice-lab.py --out ~/Desktop/<folder>/male-heart
"""
import argparse, os, sys, wave
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CLONE = HERE / "kokoro-coreml-export"
sys.path.insert(0, str(CLONE))
import torch
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

APP_PACKS = Path("/Volumes/data/Developer/schriftohr/text2Ear/Resources/KokoroVoices")
PASSAGE = ("The morning was cold and clear, and the road ran straight between the fields. "
           "Had anyone walked it before us that day? Not a footprint showed in the frost. "
           "We kept a good pace, talking of small things, until the river came into view. "
           "What a sight it was! The water lay still as glass, and the far bank was lost in mist. "
           "We stood a while, then turned for home, glad of the walk and gladder of the fire.")


def pack(name: str) -> np.ndarray:
    for p in (CLONE / "voices" / f"{name}.bin", APP_PACKS / f"{name}.bin"):
        if p.exists():
            a = np.fromfile(p, dtype=np.float32)
            assert a.size == 510 * 256, (name, a.size)
            return a.reshape(510, 256)
    raise FileNotFoundError(name)


def median_f0(x: np.ndarray, rate: int) -> float:
    win, hop = int(0.04 * rate), int(0.02 * rate)
    f0s = []
    for s in range(0, len(x) - win, hop):
        fr = x[s:s + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.02:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[win - 1:]
        ac = ac / (ac[0] + 1e-9)
        lo, hi = int(rate / 500), int(rate / 60)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.6:
            f0s.append(rate / k)
    return float(np.median(f0s)) if f0s else float("nan")


def write_wav(path: Path, x: np.ndarray, rate: int = 24000):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


class Lab:
    def __init__(self):
        cfg, ckpt = CLONE / "checkpoints/config.json", CLONE / "checkpoints/kokoro-v1_0.pth"
        # The clone's checkpoints are links to the upstream author's machine; the Hugging Face
        # cache (hexgrad/Kokoro-82M, fetched by the first export) is what actually loads.
        self.model = (KModel(config=str(cfg), model=str(ckpt)) if cfg.exists() and ckpt.exists() else KModel()).eval()
        self.pipeline = KPipeline(lang_code="a", model=self.model)
        self._f0_scale = 1.0
        original = self.model.predictor.F0Ntrain
        def scaled(en, s, _o=original):
            f0, n = _o(en, s)
            return f0 * self._f0_scale, n
        self.model.predictor.F0Ntrain = scaled

    def render(self, voice: np.ndarray, f0_scale: float = 1.0, speed: float = 1.0) -> np.ndarray:
        self._f0_scale = f0_scale
        v = torch.from_numpy(voice.astype(np.float32)).reshape(510, 1, 256)
        pieces = []
        with torch.no_grad():
            for result in self.pipeline(PASSAGE, voice=v, speed=speed):
                audio = result.audio if hasattr(result, "audio") else result[2]
                if audio is not None:
                    pieces.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
                    pieces.append(np.zeros(int(0.25 * 24000), dtype=np.float32))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def recipes():
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)          # the male direction, per row
    us_males = np.mean([pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck"]], axis=0)
    d_us = us_males - np.mean([heart, pack("af_bella"), pack("af_kore"), pack("af_nicole"), pack("af_nova"), pack("af_sarah")], axis=0)
    michael, onyx, george = pack("am_michael"), pack("am_onyx"), pack("bm_george")
    halves = lambda timbre, prosody: np.concatenate([timbre[:, :128], prosody[:, 128:]], axis=1)
    yield "heart", heart, 1.0, "Heart as she is (reference)"
    yield "onyx", onyx, 1.0, "Onyx as he is (the deepest American male pack; reference)"
    yield "heart-male100", heart + 1.0 * d, 1.0, "Heart moved one male-direction unit (all voices)"
    yield "heart-male130", heart + 1.3 * d, 1.0, "Heart moved 1.3 units toward male"
    yield "heart-male160", heart + 1.6 * d, 1.0, "Heart moved 1.6 units toward male"
    yield "heart-usmale100", heart + 1.0 * d_us, 1.0, "Heart moved one unit toward the American men only"
    yield "heart-usmale130-f090", heart + 1.3 * d_us, 0.90, "1.3 units toward the American men, pitch scaled 0.90"
    yield "heart-male100-f085", heart + 1.0 * d, 0.85, "one unit toward male, pitch scaled 0.85"
    yield "heart-prosody-michael-timbre-f060", halves(michael, heart), 0.60, "Michael's timbre, Heart's prosody, pitch scaled 0.60"
    yield "heart-prosody-onyx-timbre-f055", halves(onyx, heart), 0.55, "Onyx's timbre, Heart's prosody, pitch scaled 0.55"
    yield "heart-prosody-george-timbre-f060", halves(george, heart), 0.60, "George's timbre, Heart's prosody, pitch scaled 0.60"
    yield "heart-f060", heart, 0.60, "Heart herself, pitch scaled 0.60 (what pitch alone does)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--only", default="")
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lab = Lab()
    rows = ["| file | recipe | median pitch Hz | length s |", "|---|---|---|---|"]
    for name, voice, f0s, note in recipes():
        if a.only and name not in a.only.split(","):
            continue
        x = lab.render(voice, f0_scale=f0s)
        f0 = median_f0(x, 24000)
        write_wav(out / f"{name}.wav", x)
        row = f"| {name}.wav | {note} | {f0:.0f} | {len(x) / 24000:.1f} |"
        print("LAB " + row, flush=True); rows.append(row)
    (out / "README.md").write_text("# A male voice from Heart's style vectors, 2026-10-09\n\n"
        "Rendered by the PyTorch Kokoro (the same weights as the app), the probe passage. "
        "Speaking pitch for reference: bass about 85 to 100 Hz, baritone 100 to 125, tenor 125 to 160. "
        "Heart measured 202, Michael 121, Lewis 100.\n\n" + "\n".join(rows) + "\n")
    print("LAB done", out)
