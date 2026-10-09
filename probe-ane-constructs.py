#!/usr/bin/env python3
"""Tiny one-construct Core ML models, to ask the Neural Engine compiler which construct it refuses.

Each model holds one building block of the rewritten generator (export-ane-generator.py) at the
2 s bucket's sizes. The app's admission test (NeuralEngineAdmissionTests, TEST_RUNNER_FP16_DIR)
compiles each for the ANE and reports whether it was admitted.

    cd kokoro-coreml-export && uv run python ../probe-ane-constructs.py --out ../kokoro-coreml-ane-probes
"""
import argparse, importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kokoro-coreml-export"))
spec = importlib.util.spec_from_file_location("export_ane_generator", HERE / "export-ane-generator.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)

import torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct


class Spectrum(nn.Module):
    """The generator's end: conv_post, then exp for magnitude and sin for phase, concatenated."""
    def __init__(self):
        super().__init__()
        self.conv_post = nn.Conv1d(128, 22, 7, padding=3)
    def forward(self, x):
        y = self.conv_post(x)
        return torch.cat([torch.exp(y[:, :11]), torch.sin(y[:, 11:])], dim=1)


class Resblock(nn.Module):
    """One AdaIN-style block in phase form: instance-norm statistics plus a phase convolution."""
    def __init__(self):
        super().__init__()
        self.conv = gen.PhaseConv1d(nn.Conv1d(128, 128, 11, dilation=5, padding=25))
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True); var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x + self.conv(F.leaky_relu(x, 0.1))


class Snake(nn.Module):
    """The Snake activation as the generator writes it: x + (1/a) * sin(a x) ** 2."""
    def __init__(self, square="pow"):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 128, 1) * 0.9); self.square = square
    def forward(self, x):
        s = torch.sin(self.alpha * x)
        sq = s ** 2 if self.square == "pow" else s * s
        if self.square == "cos":
            sq = (1 - torch.cos(2 * self.alpha * x)) * 0.5
        return x + (1 / self.alpha) * sq


class SnakeResblock(nn.Module):
    """Instance norm, Snake, phase convolution: one real resblock step."""
    def __init__(self):
        super().__init__()
        self.snake = Snake(); self.conv = gen.PhaseConv1d(nn.Conv1d(128, 128, 11, dilation=5, padding=25))
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True); var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x + self.conv(self.snake(x))


def probes():
    if "--snake" in sys.argv:
        yield "snake_pow", Snake("pow"), (1, 128, 9600)
        yield "snake_mul", Snake("mul"), (1, 128, 9600)
        yield "snake_cos", Snake("cos"), (1, 128, 9600)
        yield "snake_resblock", SnakeResblock(), (1, 128, 9600)
        yield "sin_only", _Sin(), (1, 128, 9600)
        return
    torch.manual_seed(0)
    T = 9600                                    # the 2 s bucket after both upsamples
    yield "phaseconv", gen.PhaseConv1d(nn.Conv1d(128, 128, 11, dilation=5, padding=25)), (1, 128, T)
    yield "plain_dilated", nn.Conv1d(128, 128, 11, dilation=5, padding=25), (1, 128, T)
    yield "plain_k11", nn.Conv1d(128, 128, 11, padding=5), (1, 128, T)
    yield "polyup", gen.PolyphaseUpsample(nn.ConvTranspose1d(512, 256, 20, 10, padding=5)), (1, 512, 160)
    yield "zeroinsert_up", _zero_insert(nn.ConvTranspose1d(512, 256, 20, 10, padding=5)), (1, 512, 160)
    yield "polynoise", gen.PolyphaseStrided(nn.Conv1d(22, 256, 12, 6, padding=3)), (1, 22, 9601)
    yield "spectrum", Spectrum(), (1, 128, T)
    yield "resblock", Resblock(), (1, 128, T)


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def _zero_insert(conv):
    from export_synth.wrappers import ZeroInsertConvTranspose1d
    return ZeroInsertConvTranspose1d(conv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "kokoro-coreml-ane-probes"))
    ap.add_argument("--snake", action="store_true", help="the Snake activation in four forms, and plain sin")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, module, shape in probes():
        module.eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            traced = torch.jit.trace(module, x)
        model = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=shape)],
                           convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18)
        path = out / f"kokoro_probe_{name}.mlpackage"
        model.save(str(path)); print("saved", path.name)
