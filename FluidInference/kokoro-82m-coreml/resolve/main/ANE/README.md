---
license: mit
language:
  - en
library_name: coreml
pipeline_tag: text-to-speech
tags:
  - coreml
  - tts
  - kokoro
  - apple-silicon
  - ane
  - on-device
---

# Kokoro 82M — laishere CoreML port (7-stage, ANE-optimized)

CoreML conversion of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) split into a **7-stage chain** for Apple Neural Engine residency, originally produced by [@laishere](https://github.com/laishere/kokoro-coreml) (MIT). Repackaged here for use with [FluidAudio](https://github.com/FluidInference/FluidAudio).

## What's in this repo

Both `.mlpackage` (source) and `.mlmodelc` (compiled, runtime-ready) formats ship in this repo. Loaders that auto-compile (e.g. `xcrun coremlcompiler`, `MLModel.compileModel(at:)`) can use the `.mlpackage`; FluidAudio loads the `.mlmodelc` directly to skip Apple's first-run compile step.

| Stage | `.mlpackage` | `.mlmodelc` | Format | Compute target |
|---|---|---|---|---|
| `KokoroAlbert` | 5.6 MB | 5.6 MB | fp16 + int8 palettization | CPU + ANE |
| `KokoroPostAlbert` | 13 MB | 13 MB | fp16 + int8 palettization | CPU + ANE |
| `KokoroAlignment` | 20 KB | 32 KB | fp16 + int8 palettization | CPU + ANE |
| `KokoroProsody` | 8.1 MB | 8.2 MB | fp32 | CPU + GPU |
| `KokoroNoise` | 4.4 MB | 4.5 MB | fp32 | CPU + GPU |
| `KokoroVocoder` | 47 MB | 47 MB | fp16 + int8 palettization | CPU + ANE |
| `KokoroTail` | 92 KB | 100 KB | fp32 (iSTFT) | CPU + GPU |

Plus auxiliary files:

| File | Description | Size |
|---|---|---|
| `vocab.json` | 114 IPA → token IDs | 1.4 KB |
| `af_heart.bin` | flat fp32 `[510, 256]` voice pack | 512 KB |

Total: **~157 MB** with both formats (~78 MB if you keep only `.mlmodelc`, vs the original ~330 MB PyTorch weights).

## Pipeline

```
text → G2P (out-of-tree, e.g. FluidAudio's BART G2P)
     → IPA tokens [BOS, ..., EOS]   (max 512)
     → Albert        →  hidden states
     → PostAlbert    →  text features
     → Alignment     →  T_a frames (dynamic)
     → Prosody       →  pitch + duration
     → Noise         →  noise embeddings  (fp16→fp32 boundary)
     → Vocoder       →  x_pre features    (discard `anchor` output)
     → Tail (iSTFT)  →  24 kHz waveform
```

Voice pack is indexed by `row = clamp(T_enc - 1, 0, 509)`; columns `[0:128]` = timbre, `[128:256]` = style_s.

## Performance (Apple M2, 8-core)

| Stage | Steady-state |
|---|---|
| Albert | 7-10 ms |
| PostAlbert | 4-5 ms |
| Alignment | 1-2 ms |
| Prosody | 30-200 ms |
| Noise | 70-150 ms |
| Vocoder | 75-125 ms |
| Tail | 6-22 ms |

Cold model load (first run, `anecompilerservice` compilation): **~20 s**. Warm load: **~300 ms**. Steady-state RTFx: **3-11×** depending on phrase length.

## Usage with FluidAudio

```bash
swift run fluidaudiocli tts "Hello world" \
    --backend kokoro-lai \
    --output hello.wav \
    --metrics metrics.json
```

```swift
import FluidAudio

let manager = KokoroLaiManager()
try await manager.initialize()
let wav = try await manager.synthesize(text: "Hello world")
```

FluidAudio downloads this repo automatically into `~/.cache/fluidaudio/Models/kokoro-laishere/` on first use.

## Conversion

Built with [mobius/models/tts/kokoro/laishere-coreml](https://github.com/FluidInference/mobius/tree/main/models/tts/kokoro/laishere-coreml) (PyTorch 2.11 + coremltools 9.0). Reproduce:

```bash
cd mobius/models/tts/kokoro/laishere-coreml
uv sync
uv pip install --reinstall coremltools==9.0    # workaround sdist fallback
uv run python convert-coreml.py --output-dir build/laishere-kokoro
uv run python dump-benchmark-data.py --output-dir build/laishere-kokoro
for mlp in build/laishere-kokoro/Kokoro*.mlpackage; do
    xcrun coremlcompiler compile "$mlp" build/laishere-kokoro-compiled/
done
```

Parity vs PyTorch reference: waveform corr ≥ 0.80, mel-spectrogram corr ≥ 0.99 (verified by `compare-models.py`).

## Voices

This release ships only `af_heart` (American Female, "Heart"). Additional voices from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) can be re-exported by editing `dump-benchmark-data.py`'s `VOICE` constant and copying the resulting `<voice>.bin` here.

## License

MIT — inherited from upstream:
- Model weights: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0)
- CoreML conversion code + 7-stage architecture: [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml) (MIT, Lai Yongkang 2025)
- Repackaging: FluidInference (MIT)

See `LICENSE` for the upstream MIT text.

## Citation

```bibtex
@misc{kokoro-laishere-coreml,
    title  = {Kokoro 82M — 7-stage CoreML conversion for Apple Neural Engine},
    author = {Lai, Yongkang and FluidInference},
    year   = {2025},
    url    = {https://huggingface.co/FluidInference/kokoro-laishere-coreml}
}
```
