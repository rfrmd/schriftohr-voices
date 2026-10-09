#!/usr/bin/env python3
"""The decoder in overlapping windows along one sentence's own time axis: the sentence's rhythm
and pitch predicted whole, only the heavy decoder (decoder-pre + generator) run per window of at
most ten seconds, the join at the quietest frame of the overlap with a short equal-power
crossfade, no pause added. Rendered against the same sentence decoded whole, for the ear and
for the numbers. John, 2026-10-09: "cut cleanly behind a word pause ... add back pauses where needed".
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kokoro-coreml-export"))
par = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("parity", HERE / "parity-ane-generator.py")); s.loader.exec_module(par)
lab = par.lab
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

SENTENCE = ("When at last the travellers came down out of the hills into the valley, where the river ran broad "
            "and slow between the willows, they found the town already asleep; not a light showed in any window, "
            "and the only sound was the water moving under the old stone bridge.")
MAX_FRAMES = 400          # 10 s of asr frames (40 a second)
SAMPLES_PER_FRAME = 600   # 24000 / 40


def decode(model, asr, F0, N, s):
    with torch.no_grad():
        return model.decoder(asr, F0, N, s).squeeze().numpy()


def windowed(model, asr, F0, N, s, max_frames=MAX_FRAMES, fade_ms=30):
    T = asr.shape[-1]
    if T <= max_frames:
        return decode(model, asr, F0, N, s), []
    # The fewest windows that cover the sentence with at least two seconds of overlap, evenly spaced.
    overlap = 80
    k = int(np.ceil((T - overlap) / (max_frames - overlap)))
    starts = [int(round(x)) for x in np.linspace(0, T - max_frames, k)]
    out = None; seams = []
    for a in starts:
        b = min(a + max_frames, T)
        piece = decode(model, asr[:, :, a:b], F0[:, 2 * a:2 * b], N[:, 2 * a:2 * b], s)
        off = a * SAMPLES_PER_FRAME
        if out is None:
            out = piece; continue
        # Join inside the overlap at the quietest 20 ms of what we already have, with an equal-power crossfade.
        lo, hi = off + int(0.3 * 24000), len(out) - int(0.3 * 24000)
        hop = 480
        energies = [(np.mean(out[i:i + hop] ** 2), i) for i in range(lo, hi - hop, hop)]
        cut = min(energies)[1] + hop // 2
        n = int(fade_ms / 1000 * 24000)
        t = np.linspace(0, 1, n, dtype=np.float32)
        fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
        head = out[:cut - n // 2].copy()
        mixed = out[cut - n // 2:cut + n // 2] * fade_out + piece[cut - n // 2 - off:cut + n // 2 - off] * fade_in
        tail = piece[cut + n // 2 - off:]
        out = np.concatenate([head, mixed, tail]); seams.append(cut / 24000)
    return out, seams


def main(out_dir):
    out = Path(out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    cfg, ckpt = par.CLONE / "checkpoints/config.json", par.CLONE / "checkpoints/kokoro-v1_0.pth"
    model = KModel(config=str(cfg), model=str(ckpt), disable_complex=True).eval()
    pipeline = KPipeline(lang_code="a", model=model)
    heart = lab.pack("af_heart")
    males = [lab.pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [lab.pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    john = heart + 1.6 * (np.mean(males, axis=0) - np.mean(females, axis=0))
    rows = ["| voice | whole s | windows | seams at s | envelope corr | mean spectral diff dB | spectral diff in the 0.5 s round each seam |", "|---|---|---|---|---|---|---|"]
    for vname, pack in (("heart", heart), ("john", john)):
        ps, vi, reference = par.internals(model, pipeline, SENTENCE, pack)
        asr, F0, N = (torch.from_numpy(vi[k]) for k in ("asr", "f0_curve", "n"))
        s = torch.from_numpy(vi["ref_s"])[:, :128]
        T = asr.shape[-1]
        win, seams = windowed(model, asr, F0, N, s)
        n = min(len(reference), len(win)); r, w = reference[:n], win[:n]
        # Each window's harmonic source starts its phase afresh, so sample correlation is blind here;
        # the envelope (RMS per 10 ms) and the log spectrum (per 20 ms) carry what the ear hears.
        hop = 240
        env = lambda x: np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
        er, ew = env(r), env(w)
        corr = np.corrcoef(er, ew)[0, 1]
        frame = 480
        def logspec(x):
            frames = [x[i:i + frame] * np.hanning(frame) for i in range(0, len(x) - frame, frame // 2)]
            return np.log10(np.abs(np.fft.rfft(np.array(frames), axis=1)) + 1e-6)
        lr, lw = logspec(r), logspec(w)
        snr = float(np.mean(np.abs(lr - lw)) * 20)          # mean spectral difference in dB
        local = []
        for t in seams:
            i0, i1 = int((t - 0.25) * 24000), int((t + 0.25) * 24000)
            lr2, lw2 = logspec(r[i0:i1]), logspec(w[i0:i1])
            local.append(float(np.mean(np.abs(lr2 - lw2)) * 20))
        lab.write_wav(out / f"{vname}-whole.wav", reference); lab.write_wav(out / f"{vname}-windowed.wav", win)
        row = f"| {vname} | {len(reference)/24000:.1f} | {len(seams)+1} of {T} frames max {MAX_FRAMES} | {', '.join(f'{t:.2f}' for t in seams)} | {corr:.4f} | {snr:.1f} | {', '.join(f'{x:.1f}' for x in local)} |"
        print("WIN " + row); rows.append(row)
    (out / "README.md").write_text("# One 15 s sentence decoded whole and in two overlapping windows, 2026-10-09\n\n" + SENTENCE +
        "\n\nThe rhythm and pitch are predicted once for the whole sentence; only the decoder runs per window (at most 10 s), joined at the quietest 20 ms of the overlap with a 30 ms equal-power crossfade; no pause added.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
