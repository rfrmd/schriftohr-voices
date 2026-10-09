#!/usr/bin/env python3
"""Does the Neural Engine generator sound like the model? Real sentence inputs, the fp16 stage
run through Core ML on this Mac's Neural Engine, the inverse STFT applied, and the waveform
compared with PyTorch's own decoder. Writes both as WAV so the stage can be heard.

    cd kokoro-coreml-export && uv run python ../parity-ane-generator.py --packages ../kokoro-coreml-ane-full --out <folder>
"""
import argparse, importlib.util, sys, wave
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CLONE = HERE / "kokoro-coreml-export"
sys.path.insert(0, str(CLONE))
import torch
import coremltools as ct
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np

spec = importlib.util.spec_from_file_location("voice_lab", HERE / "voice-lab.py")
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)


def write_wav(path, x, rate=24000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def internals(model, pipeline, text, voice_pack, speed=1.0):
    """asr, F0, N, ref_s for one piece of text (the first chunk), as the model computes them."""
    for result in pipeline(text, voice=torch.from_numpy(voice_pack).reshape(510, 1, 256), speed=speed):
        ps = result.phonemes
        break
    ids = [model.vocab[p] for p in ps if p in model.vocab]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = torch.from_numpy(voice_pack[len(ps) - 1:len(ps)].astype(np.float32))        # [1, 256]
    with torch.no_grad():
        lengths = torch.full((1,), input_ids.shape[-1], dtype=torch.long)
        text_mask = torch.gt(torch.arange(lengths.max()).unsqueeze(0) + 1, lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        aln = torch.zeros((input_ids.shape[1], indices.shape[0])); aln[indices, torch.arange(indices.shape[0])] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, lengths, text_mask)
        asr = t_en @ aln
        reference = model.decoder(asr, F0, N, ref_s[:, :128]).squeeze().numpy()
    return ps, {"asr": asr.numpy(), "f0_curve": F0.numpy(), "n": N.numpy(), "ref_s": ref_s.numpy()}, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", default=str(HERE / "kokoro-coreml-ane-full"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The water lay still as glass, and the far bank was lost in mist.")
    ap.add_argument("--units", default="CPU_AND_NE", choices=["CPU_AND_NE", "CPU_ONLY", "ALL"])
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    cfg, ckpt = CLONE / "checkpoints/config.json", CLONE / "checkpoints/kokoro-v1_0.pth"
    model = KModel(config=str(cfg), model=str(ckpt), disable_complex=True).eval()
    pipeline = KPipeline(lang_code="a", model=model)
    gen = model.decoder.generator
    heart = lab.pack("af_heart")
    males = [lab.pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
    females = [heart] + [lab.pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
    direction = np.mean(males, axis=0) - np.mean(females, axis=0)
    voices = {"heart": heart, "heart-male160": heart + 1.6 * direction}
    units = getattr(ct.ComputeUnit, a.units)
    rows = ["| voice | bucket | reference s | corr | SNR dB | max diff / max ref | stage ms |", "|---|---|---|---|---|---|---|"]
    for vname, vpack in voices.items():
        ps, vi, reference = internals(model, pipeline, a.text, vpack)
        seconds = len(reference) / 24000
        bucket = next(b for b in (2, 3, 7, 10) if b >= seconds + 0.3)
        path = Path(a.packages) / f"kokoro_decoder_har_post_{bucket}s.mlpackage"
        mlmodel = ct.models.MLModel(str(path), compute_units=units)
        desc = mlmodel.get_spec().description
        shapes = {i.name: list(i.type.multiArrayType.shape) for i in desc.input}
        x_pre, ref_s, har, _, _, mask = build_decoder_har_post_inputs_np(model.decoder, vi, bucket, shapes["x_pre"][-1], shapes["har"][-1], warn_geometry=False)
        feed = {"x_pre": x_pre, "ref_s": ref_s, "har": har}
        if "mask" in shapes: feed["mask"] = mask
        import time
        mlmodel.predict(feed)                                       # warm
        t0 = time.perf_counter(); outputs = mlmodel.predict(feed); ms = (time.perf_counter() - t0) * 1000
        y = torch.from_numpy(next(iter(outputs.values())).astype(np.float32))
        if y.ndim == 3 and y.shape[1] == 22:
            with torch.no_grad():
                wave_out = gen.stft.inverse(y[:, :11], y[:, 11:]).squeeze().numpy()
        else:
            wave_out = y.squeeze().numpy()
        n = min(len(reference), len(wave_out))
        r, w = reference[:n], wave_out[:n]
        corr = float(np.corrcoef(r, w)[0, 1]); snr = 10 * np.log10(np.sum(r ** 2) / (np.sum((r - w) ** 2) + 1e-12))
        row = f"| {vname} | {bucket} s | {seconds:.2f} | {corr:.5f} | {snr:.1f} | {np.max(np.abs(r - w)) / np.max(np.abs(r)):.4f} | {ms:.0f} |"
        print("PARITY " + row, flush=True); rows.append(row)
        write_wav(out / f"{vname}-pytorch.wav", reference); write_wav(out / f"{vname}-ane-{bucket}s.wav", wave_out[:len(reference)])
    (out / "README.md").write_text("# The Neural Engine generator against PyTorch, one sentence\n\n" + f"Text: {a.text}\n\nCompute units {a.units} on this Mac; the stage's output finished by the model's own inverse STFT.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
