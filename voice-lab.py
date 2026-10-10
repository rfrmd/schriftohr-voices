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


def refine():
    """John's ear (2026-10-09, 01:40): heart-male160 ("John") and Michael's timbre with Heart's
    prosody at pitch 0.60 ("Jeremiah") are both good; refine a tad lower toward bass."""
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)
    michael = pack("am_michael")
    halves = lambda timbre, prosody: np.concatenate([timbre[:, :128], prosody[:, 128:]], axis=1)
    john = heart + 1.6 * d
    jeremiah = halves(michael, heart)
    yield "john", john, 1.0, "John as heard: Heart moved 1.6 units toward male"
    yield "john-f092", john, 0.92, "John, pitch scaled 0.92"
    yield "john-f085", john, 0.85, "John, pitch scaled 0.85"
    yield "john-180", heart + 1.8 * d, 1.0, "Heart moved 1.8 units toward male"
    yield "john-180-f092", heart + 1.8 * d, 0.92, "1.8 units, pitch scaled 0.92"
    yield "john-200", heart + 2.0 * d, 1.0, "Heart moved 2.0 units toward male"
    yield "jeremiah", jeremiah, 0.60, "Jeremiah as heard: Michael's timbre, Heart's prosody, pitch 0.60"
    yield "jeremiah-f054", jeremiah, 0.54, "Jeremiah, pitch scaled 0.54"
    yield "jeremiah-f048", jeremiah, 0.48, "Jeremiah, pitch scaled 0.48"
    yield "jeremiah-deeper-timbre-f056", halves(michael + 0.5 * d, heart), 0.56, "Michael's timbre moved half a unit further male, Heart's prosody, pitch 0.56"
    yield "jeremiah-deeper-timbre-f050", halves(michael + 0.5 * d, heart), 0.50, "the same timbre, pitch 0.50"


def refine2():
    """Jeremiah with the pitch lowered through the style vector alone (Heart's prosody half
    moved along the male direction), so the voice is a plain pack and needs no pitch knob."""
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)
    michael = pack("am_michael")
    halves = lambda timbre, prosody: np.concatenate([timbre[:, :128], prosody[:, 128:]], axis=1)
    for lam in (1.4, 1.6, 1.8, 2.0):
        yield f"jeremiah-style{int(lam*100)}", halves(michael, heart + lam * d), 1.0, f"Michael's timbre, Heart's prosody moved {lam} units toward male, no pitch scaling"
    yield "jeremiah-style160-timbre-plus", halves(michael + 0.5 * d, heart + 1.6 * d), 1.0, "Michael's timbre half a unit further male, Heart's prosody moved 1.6 units"
    yield "john-160", heart + 1.6 * d, 1.0, "John (1.6 units), the pack as it would ship"
    yield "john-180", heart + 1.8 * d, 1.0, "John at 1.8 units, the pack as it would ship"


def refine3():
    """Jeremiah's TONE moved off Michael's (John, 2026-10-09: "in listening to jeremiah and michael
    the tone sounds almost the same"). The shipped Jeremiah keeps Michael's timbre half whole, so its
    tone is Michael's by construction; these move the timbre half while keeping the prosody half
    (Heart moved 2.0 units toward male) that John chose."""
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)
    michael, onyx, lewis, george = pack("am_michael"), pack("am_onyx"), pack("bm_lewis"), pack("bm_george")
    halves = lambda timbre, prosody: np.concatenate([timbre[:, :128], prosody[:, 128:]], axis=1)
    prosody = heart + 2.0 * d
    yield "jeremiah-as-shipped", halves(michael, prosody), 1.0, "the shipped Jeremiah: Michael's timbre, Heart's prosody at 2.0 units"
    yield "jeremiah-t1", halves(michael + 1.0 * d, prosody), 1.0, "Michael's timbre moved a full unit further male"
    yield "jeremiah-t2", halves(michael + 2.0 * d, prosody), 1.0, "Michael's timbre moved two units further male"
    yield "jeremiah-t3", halves(0.5 * michael + 0.5 * (heart + 2.0 * d), prosody), 1.0, "timbre halfway between Michael and Heart-moved-male (John's timbre)"
    yield "jeremiah-t4", halves(0.5 * michael + 0.5 * onyx, prosody), 1.0, "timbre halfway between Michael and Onyx, the pack's bass-baritone"
    yield "jeremiah-t5", halves(onyx, prosody), 1.0, "Onyx's timbre whole, Heart's prosody at 2.0 units"
    yield "jeremiah-t6", halves(0.5 * michael + 0.5 * lewis, prosody), 1.0, "timbre halfway between Michael and Lewis"
    yield "jeremiah-t7", halves(george, prosody), 1.0, "George's timbre whole, Heart's prosody at 2.0 units"
    yield "jeremiah-t8", halves(heart + 3.0 * d, prosody), 1.0, "Heart's own timbre moved three units male (no Michael at all)"
    yield "michael-as-shipped", michael, 1.0, "Michael, for the comparison"


def john2():
    """John revision 2 (John, 2026-10-10: the shipped John carries "a very slight buzz to the voice"
    on AirPods Pro, absent from the original voices; goal "Heart's speech pattern in a deeper timbre
    without the buzz"). The shipped John moves BOTH halves 1.6 units toward male; the decoder's
    timbre half, pushed that far off every real voice, is the likely source of the buzz. These keep
    John's prosody half exactly (Heart's pattern, moved male) and bring the timbre half back toward
    real voices; the last one moves the prosody instead, to show which half carries the buzz."""
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)
    michael, adam, liam = pack("am_michael"), pack("am_adam"), pack("am_liam")
    halves = lambda timbre, prosody: np.concatenate([timbre[:, :128], prosody[:, 128:]], axis=1)
    john = heart + 1.6 * d
    yield "john-as-shipped", john, 1.0, "the shipped John: Heart moved 1.6 units toward male, both halves"
    yield "john2-t12", halves(heart + 1.2 * d, john), 1.0, "timbre pulled back to 1.2 units, John's prosody"
    yield "john2-t14", halves(heart + 1.4 * d, john), 1.0, "timbre pulled back to 1.4 units, John's prosody"
    yield "john2-m30", halves(0.7 * (heart + 1.6 * d) + 0.3 * michael, john), 1.0, "John's timbre blended 30% toward Michael's"
    yield "john2-m50", halves(0.5 * (heart + 1.6 * d) + 0.5 * michael, john), 1.0, "John's timbre blended 50% toward Michael's"
    yield "john2-hm", halves(0.5 * heart + 0.5 * michael, john), 1.0, "timbre halfway between Heart and Michael (no extrapolation at all)"
    yield "john2-ha", halves(0.4 * heart + 0.6 * adam, john), 1.0, "timbre 40% Heart, 60% Adam (no extrapolation)"
    yield "john2-hl", halves(0.4 * heart + 0.6 * liam, john), 1.0, "timbre 40% Heart, 60% Liam (no extrapolation)"
    yield "john2-p12", halves(john, heart + 1.2 * d), 1.0, "diagnostic: John's timbre kept, prosody pulled back to 1.2 units"


def pace():
    """John's ear (02:00): John has the cleanest pitch, John at 0.85 the better pace. The pitch
    scale changes no timing; this is the real pace knob (Kokoro's speed) on John."""
    heart = pack("af_heart")
    males = [pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
                               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    d = np.mean(males, axis=0) - np.mean(females, axis=0)
    john = heart + 1.6 * d
    yield "john-speed095", john, 1.0, 0.95, "John, pace 0.95"
    yield "john-speed090", john, 1.0, 0.90, "John, pace 0.90"
    yield "john-f092-speed095", john, 0.92, 0.95, "John, pitch 0.92, pace 0.95"
    yield "john-f085-speed100", john, 0.85, 1.00, "John, pitch 0.85 (the one heard), pace 1.0, for reference"


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
    ap.add_argument("--refine", action="store_true", help="the John and Jeremiah refinements")
    ap.add_argument("--refine2", action="store_true", help="Jeremiah lowered through the style vector alone; the packs as they would ship")
    ap.add_argument("--refine3", action="store_true", help="Jeremiah's tone moved off Michael's: the timbre half varied, the prosody kept")
    ap.add_argument("--pace", action="store_true", help="John at slower paces")
    ap.add_argument("--john2", action="store_true", help="John revision 2: the timbre half brought back toward real voices, the prosody kept")
    ap.add_argument("--packs", default="", help="write each recipe's [510, 256] pack as <name>.bin into this folder, with SHA256SUMS")
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lab = Lab()
    rows = ["| file | recipe | median pitch Hz | length s |", "|---|---|---|---|"]
    import hashlib
    packs_dir = Path(a.packs).expanduser() if a.packs else None
    if packs_dir: packs_dir.mkdir(parents=True, exist_ok=True); sums = []
    items = [(n, v, f, 1.0, note) for n, v, f, note in (john2() if a.john2 else refine3() if a.refine3 else refine2() if a.refine2 else refine() if a.refine else recipes())] if not a.pace else list(pace())
    for name, voice, f0s, speed, note in items:
        if packs_dir and f0s == 1.0:
            raw = voice.astype(np.float32).tobytes(); (packs_dir / f"{name}.bin").write_bytes(raw)
            sums.append(f"{hashlib.sha256(raw).hexdigest()}  {name}.bin")
            (packs_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        if a.only and name not in a.only.split(","):
            continue
        x = lab.render(voice, f0_scale=f0s, speed=speed)
        f0 = median_f0(x, 24000)
        write_wav(out / f"{name}.wav", x)
        row = f"| {name}.wav | {note} | {f0:.0f} | {len(x) / 24000:.1f} |"
        print("LAB " + row, flush=True); rows.append(row)
    (out / "README.md").write_text(("# John at slower paces, 2026-10-09\n\n" if a.pace else ("# John revision 2: the timbre half brought back, 2026-10-10\n\n" if a.john2 else "# Jeremiah's tone moved off Michael's, 2026-10-09\n\n" if a.refine3 else "# Jeremiah by style alone, and the packs as they would ship, 2026-10-09\n\n") if a.refine2 else "# John and Jeremiah, refined toward bass, 2026-10-09\n\n" if a.refine else "# A male voice from Heart's style vectors, 2026-10-09\n\n") +
        "Rendered by the PyTorch Kokoro (the same weights as the app), the probe passage. "
        "Speaking pitch for reference: bass about 85 to 100 Hz, baritone 100 to 125, tenor 125 to 160. "
        "Heart measured 202, Michael 121, Lewis 100.\n\n" + "\n".join(rows) + "\n")
    print("LAB done", out)
