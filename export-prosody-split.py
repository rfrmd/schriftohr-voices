#!/usr/bin/env python3
"""Export the halves of Kokoro's two processor stages that the Neural Engine could take whole.

The duration model (kokoro_duration_t256) and the pitch model (kokoro_f0ntrain_t*) run on the
processor in both routes because each carries LSTMs, which the Neural Engine compiler will not
take, and a stage that runs partly on each unit is the OS 27 crash class. But most of each
stage's arithmetic is not LSTM:

  duration  = ALBERT (12 shared layers, 256 tokens)  +  DurationEncoder LSTMs, duration LSTM,
              text-encoder convs+LSTM, projection
  f0ntrain  = shared bi-LSTM  +  F0 / N AdaIN residual conv stacks, 2x upsample, projections

This script exports each stage's non-recurrent body on its own, in half precision, as the
Neural Engine would need it, plus the recurrent remainder in full precision for the processor:

  kokoro_bert_t256            input_ids, attention_mask  -> d_en [1,512,256]   (gathers inside)
  kokoro_bert_emb_t256        word_emb [1,256,128], mask [1,256] -> d_en      (no gather, no int op:
                              the app looks the 128-wide word embeddings up itself; position and
                              token-type embeddings are folded into one constant)
  kokoro_f0lstm_t<T>          en [1,640,T] -> x [1,512,T]                       (processor, fp32)
  kokoro_f0convs_t<T>         x [1,512,T], s [1,128] -> F0 [1,2T], N [1,2T]     (Neural Engine, fp16)

It also writes the word-embedding table (kokoro_word_embeddings.bin, 178 x 128 float32) and
checks each Core ML package against PyTorch on the Mac. Nothing here touches the upstream clone
or the shipped sets; the app decides later whether to carry any of it.

Weights come from kokoro-weights/ (hashes in SHA256SUMS) through the clone's checkpoints/ links.
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLONE = HERE / "kokoro-coreml-export"
sys.path.insert(0, str(CLONE))
os.chdir(CLONE)   # checkpoints/ is resolved relative to the clone

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from export_synth import convert as cv

T_TOKENS = 256


class BertHalf(nn.Module):
    """The ALBERT body as upstream traces it inside the duration model, ending at d_en."""
    def __init__(self, k):
        super().__init__()
        self.bert, self.enc = k.bert, k.bert_encoder
        if hasattr(self.bert.embeddings, "token_type_ids"):
            delattr(self.bert.embeddings, "token_type_ids")

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=torch.zeros_like(input_ids))
        return self.enc(h).transpose(-1, -2)


class BertHalfEmbedded(nn.Module):
    """The same body with the embedding lookup outside: no gather, no integer anywhere."""
    def __init__(self, k):
        super().__init__()
        e = k.bert.embeddings
        pos_type = e.position_embeddings.weight[:T_TOKENS] + e.token_type_embeddings.weight[0]
        self.register_buffer("pos_type", pos_type.detach().unsqueeze(0).clone())
        self.ln = e.LayerNorm
        self.encoder = k.bert.encoder
        self.enc = k.bert_encoder
        self.layers = k.bert.config.num_hidden_layers

    def forward(self, word_emb, mask):
        x = self.ln(word_emb + self.pos_type)
        ext = (1.0 - mask)[:, None, None, :] * -1.0e4   # fp16-safe "minus infinity"
        out = self.encoder(x, attention_mask=ext, head_mask=[None] * self.layers)
        h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        return self.enc(h).transpose(-1, -2)


class F0Lstm(nn.Module):
    """The pitch stage's shared bi-LSTM: the half that stays on the processor."""
    def __init__(self, k):
        super().__init__()
        self.shared = k.predictor.shared

    def forward(self, en):                    # [1, 640, T]
        x, _ = self.shared(en.transpose(-1, -2))
        return x.transpose(-1, -2)            # [1, 512, T]


class F0Convs(nn.Module):
    """Everything after the LSTM: the AdaIN residual stacks and projections, F0 and N."""
    def __init__(self, k):
        super().__init__()
        p = k.predictor
        self.F0, self.N, self.F0_proj, self.N_proj = p.F0, p.N, p.F0_proj, p.N_proj

    def forward(self, x, s):                  # x [1, 512, T], s [1, 128]
        f0, n = x, x
        for b in self.F0: f0 = b(f0, s)
        for b in self.N: n = b(n, s)
        return self.F0_proj(f0).squeeze(1), self.N_proj(n).squeeze(1)


def convert(module, name, inputs, outputs, precision, out_dir, example):
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
    cp = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    t0 = time.time()
    ml = ct.convert(traced, inputs=inputs, outputs=outputs, convert_to="mlprogram",
                    minimum_deployment_target=ct.target.iOS18, compute_precision=cp,
                    compute_units=ct.ComputeUnit.ALL)
    path = out_dir / f"{name}.mlpackage"
    if path.exists():
        import shutil; shutil.rmtree(path)
    ml.save(str(path))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"{name}: converted in {time.time() - t0:.0f} s, {size:.1f} MB, {precision}")
    return ml, traced


def check(ml, module, feed, names, label):
    """Max abs / rel error of the Core ML package against PyTorch, and best-of-5 ms on this Mac."""
    with torch.no_grad():
        ref = module(*[torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in feed.values()])
    refs = list(ref) if isinstance(ref, (tuple, list)) else [ref]
    got = ml.predict(feed)
    _ = ml.predict(feed)
    best = 1e9
    for _ in range(5):
        t0 = time.time(); ml.predict(feed); best = min(best, (time.time() - t0) * 1000)
    parts = []
    for n, r in zip(names, refs):
        g = np.asarray(got[n], dtype=np.float32).reshape(-1); rr = r.numpy().astype(np.float32).reshape(-1)
        err = np.abs(g - rr).max(); rel = err / (np.abs(rr).max() + 1e-9)
        parts.append(f"{n}: max|err| {err:.4g} ({rel * 100:.3f}% of range)")
    print(f"  {label}: {best:.1f} ms on this Mac (coremltools, all units); " + "; ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="400", help="pitch buckets to export, in frames (t), comma separated")
    ap.add_argument("--out", default=str(HERE / "kokoro-coreml-prosody-split"))
    ap.add_argument("--inputs", default=str(HERE / "ane-probe-inputs" / "inputs-duration"),
                    help="input_ids_short.bin / attention_mask_short.bin for the checks")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    k = cv.prepare_pytorch_models("checkpoints/config.json", "checkpoints/kokoro-v1_0.pth").eval()
    emb = k.bert.embeddings.word_embeddings.weight.detach().numpy().astype(np.float32)
    emb.tofile(out / "kokoro_word_embeddings.bin")
    print(f"word embeddings {emb.shape} -> kokoro_word_embeddings.bin")

    ids = np.fromfile(Path(a.inputs) / "input_ids_short.bin", dtype=np.int32).reshape(1, T_TOKENS)
    mask = np.fromfile(Path(a.inputs) / "attention_mask_short.bin", dtype=np.int32).reshape(1, T_TOKENS)
    ids_t, mask_t = torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(mask.astype(np.int64))

    # 1. ALBERT with its gathers.
    bert = BertHalf(k)
    ml, _ = convert(bert, "kokoro_bert_t256",
                    [ct.TensorType(name="input_ids", shape=(1, T_TOKENS), dtype=np.int32),
                     ct.TensorType(name="attention_mask", shape=(1, T_TOKENS), dtype=np.int32)],
                    [ct.TensorType(name="d_en")], "float16", out, (ids_t, mask_t))
    check(ml, bert, {"input_ids": ids, "attention_mask": mask}, ["d_en"], "kokoro_bert_t256 fp16")

    # 2. ALBERT with the lookup outside.
    bert_e = BertHalfEmbedded(k)
    word_emb = emb[ids[0]][None].astype(np.float32)                 # [1, 256, 128]
    maskf = mask.astype(np.float32)
    with torch.no_grad():
        same = (bert_e(torch.from_numpy(word_emb), torch.from_numpy(maskf)) - bert(ids_t, mask_t)).abs().max().item()
    print(f"embedded body vs whole body in PyTorch, max|diff| {same:.3g} (valid tokens only matter; pads differ by the mask value)")
    ml, _ = convert(bert_e, "kokoro_bert_emb_t256",
                    [ct.TensorType(name="word_emb", shape=(1, T_TOKENS, 128), dtype=np.float32),
                     ct.TensorType(name="mask", shape=(1, T_TOKENS), dtype=np.float32)],
                    [ct.TensorType(name="d_en")], "float16", out,
                    (torch.from_numpy(word_emb), torch.from_numpy(maskf)))
    check(ml, bert_e, {"word_emb": word_emb, "mask": maskf}, ["d_en"], "kokoro_bert_emb_t256 fp16")

    # 3. The pitch stage's two halves, per bucket.
    for t in [int(x) for x in a.frames.split(",")]:
        en = torch.randn(1, 640, t) * 0.5
        s = torch.randn(1, 128) * 0.3
        lstm = F0Lstm(k)
        ml, _ = convert(lstm, f"kokoro_f0lstm_t{t}", [ct.TensorType(name="en", shape=(1, 640, t), dtype=np.float32)],
                        [ct.TensorType(name="x")], "float32", out, (en,))
        check(ml, lstm, {"en": en.numpy()}, ["x"], f"kokoro_f0lstm_t{t} fp32")
        with torch.no_grad():
            x = lstm(en)
        convs = F0Convs(k)
        ml, _ = convert(convs, f"kokoro_f0convs_t{t}",
                        [ct.TensorType(name="x", shape=(1, 512, t), dtype=np.float32),
                         ct.TensorType(name="s", shape=(1, 128), dtype=np.float32)],
                        [ct.TensorType(name="F0"), ct.TensorType(name="N")], "float16", out, (x, s))
        check(ml, convs, {"x": x.numpy(), "s": s.numpy()}, ["F0", "N"], f"kokoro_f0convs_t{t} fp16")
        # Inputs for the device probe: the LSTM's real output for this bucket and a style.
        probe = HERE / "ane-probe-inputs" / f"inputs-f0-t{t}"
        probe.mkdir(parents=True, exist_ok=True)
        x.numpy().astype(np.float32).tofile(probe / "x.bin"); s.numpy().astype(np.float32).tofile(probe / "s.bin")
        en.numpy().astype(np.float32).tofile(probe / "en.bin")
    print("done ->", out)


if __name__ == "__main__":
    main()
