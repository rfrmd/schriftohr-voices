#!/usr/bin/env python3
"""Export Kokoro's generator for the Neural Engine, without touching the upstream clone.

The ANE compiler refuses the generator as upstream exports it: a dilated 1-D convolution is a
"large kernel" to it (kernel 7 at dilation 5 spans 31 taps, kernel 11 at dilation 5 spans 51),
and large kernels must span a multiple of 8. This replaces every dilated Conv1d in the
generator with the same arithmetic in a different shape: the time axis is split into d
phases ([B, C, T] → [B, C, T/d, d]) and an ordinary Conv2d with kernel (k, 1) runs along
each phase. Output is identical to the dilated convolution; the ANE sees a small 2-D kernel.
The two upsample ConvTranspose1d layers (kernels 20 and 12, strides 10 and 6) are the next refusal:
upstream's zero-insert rewrite keeps the 20-tap kernel. Here each becomes its exact polyphase form,
a 3-tap Conv1d producing u times the channels followed by a reshape (depth to space). The first
noise convolution (kernel 12, stride 6) becomes a reshape (space to depth) and a 3-tap Conv1d.
With --no-istft the stage ends at the spectrum and Swift finishes the inverse STFT.

    cd kokoro-coreml-export && uv run python ../export-ane-generator.py --buckets 3s --out ../kokoro-coreml-ane
"""
import argparse, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLONE = HERE / "kokoro-coreml-export"
sys.path.insert(0, str(CLONE))

import torch
import torch.nn as nn
import torch.nn.functional as F
from export_synth import convert as cv


class PhaseConv1d(nn.Module):
    """A dilated Conv1d as a Conv2d over (time / d, phase): the same numbers, no dilation."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if hasattr(conv, "weight_g") or hasattr(conv, "parametrizations"):
            try:
                torch.nn.utils.remove_weight_norm(conv)
            except Exception:
                torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
        self.k = conv.kernel_size[0]
        self.d = conv.dilation[0]
        self.p = conv.padding[0]
        assert conv.stride[0] == 1 and conv.groups == 1
        assert self.p == (self.k - 1) * self.d // 2, (self.k, self.d, self.p)
        self.weight = nn.Parameter(conv.weight.detach().clone().unsqueeze(-1))   # [Cout, Cin, k, 1]
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

    def forward(self, x):
        b, c, t = x.shape
        tp = t + 2 * self.p
        extra = (-tp) % self.d
        x = F.pad(x, (self.p, self.p + extra))
        q = (tp + extra) // self.d
        x = x.reshape(b, c, q, self.d)
        y = F.conv2d(x, self.weight, self.bias)             # [B, Cout, q - k + 1, d]
        y = y.reshape(b, y.shape[1], (q - self.k + 1) * self.d)
        return y[:, :, :t]


def rewrite_dilated(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d) and child.dilation[0] > 1 and child.kernel_size[0] > 1:
            setattr(module, name, PhaseConv1d(child))
            n += 1
        else:
            n += rewrite_dilated(child)
    return n


def _plain(conv):
    if hasattr(conv, "weight_g") or hasattr(conv, "parametrizations"):
        try:
            torch.nn.utils.remove_weight_norm(conv)
        except Exception:
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
    return conv


class PolyphaseUpsample(nn.Module):
    """ConvTranspose1d(k, stride u, padding p) with k - 2p = u, as a 3-tap Conv1d and a reshape.

    Output t = q*u + r takes input i = q + s, s in [smin, smax], with weight tap
    j = -s*u + r + p. Channel o*u + r of the conv is phase r of output channel o.
    """

    def __init__(self, conv: nn.ConvTranspose1d):
        super().__init__()
        conv = _plain(conv)
        k, u, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        assert k - 2 * p == u and conv.output_padding[0] == 0 and conv.groups == 1, (k, u, p)
        cin, cout = conv.in_channels, conv.out_channels
        dmax = (k - 1 - p) // u                       # largest q - i
        dmin = -((u - 1 + p) // u)                    # smallest q - i (ceil of a negative)
        self.u, self.cout = u, cout
        self.pad_left, self.pad_right = dmax, -dmin   # i = q - dmax .. q - dmin
        taps = dmax - dmin + 1
        w = conv.weight.detach()                      # [Cin, Cout, k]
        weight = torch.zeros(cout * u, cin, taps)
        for m in range(taps):
            d = dmax - m                              # q - i for tap m (i ascending)
            for r in range(u):
                j = d * u + r + p
                if 0 <= j < k:
                    weight[torch.arange(cout) * u + r, :, m] = w[:, :, j].t()
        self.weight = nn.Parameter(weight)
        bias = conv.bias.detach() if conv.bias is not None else torch.zeros(cout)
        self.bias = nn.Parameter(bias.repeat_interleave(u))

    def forward(self, x):
        b, _, L = x.shape
        y = F.conv1d(F.pad(x, (self.pad_left, self.pad_right)), self.weight, self.bias)   # [B, Cout*u, L]
        return y.reshape(b, self.cout, self.u, L).transpose(2, 3).reshape(b, self.cout, L * self.u)


class PolyphaseStrided(nn.Module):
    """Conv1d(k, stride s, padding p) as a reshape (space to depth) and a short Conv1d."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        conv = _plain(conv)
        k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        assert conv.dilation[0] == 1 and conv.groups == 1
        cin, cout = conv.in_channels, conv.out_channels
        amax = (k - 1 - p) // s
        amin = -((s - 1 + p) // s)
        self.k, self.s, self.p, self.cin = k, s, p, cin
        self.pad_left, self.pad_right = -amin, amax
        taps = amax - amin + 1
        w = conv.weight.detach()                      # [Cout, Cin, k]
        weight = torch.zeros(cout, cin * s, taps)
        for m in range(taps):
            a = amin + m
            for r in range(s):
                j = a * s + r + p
                if 0 <= j < k:
                    weight[:, torch.arange(cin) * s + r, m] = w[:, :, j]
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

    def forward(self, x):
        b, c, t = x.shape
        out_len = (t + 2 * self.p - self.k) // self.s + 1
        extra = (-t) % self.s
        x = F.pad(x, (0, extra)).reshape(b, c, (t + extra) // self.s, self.s)
        x = x.permute(0, 1, 3, 2).reshape(b, c * self.s, (t + extra) // self.s)
        y = F.conv1d(F.pad(x, (self.pad_left, self.pad_right)), self.weight, self.bias)
        return y[:, :, :out_len]


class SliceReflectPad(nn.Module):
    """ReflectionPad1d((1, 0)) as a slice and a concat: the same frame, no pad op.
    The Neural Engine compiler refuses the generator with the reflection pad in it
    (found by bisection, 2026-10-09) and takes it with this."""

    def forward(self, x):
        return torch.cat([x[:, :, 1:2], x], dim=2)


def _snake(x, alpha):
    """x + sin(alpha x)^2 / alpha, with 1/alpha split into two fp16-sized factors when it is
    too large for fp16 (noise_res[1].alpha2[0] has 1/alpha above 65504, so coremltools kept
    that one multiply at fp32 and the Neural Engine handed it to the processor)."""
    inv = 1.0 / alpha
    peak = float(inv.abs().max())
    if peak <= 60000.0:
        return x + inv * (torch.sin(alpha * x) ** 2)
    scale = 2.0 ** math.ceil(math.log2(peak / 60000.0))
    return x + ((inv / scale) * (torch.sin(alpha * x) ** 2)) * scale


def _resblock_forward(self, x, s, m=None):
    """AdaINResBlock1.forward as upstream writes it, with the Snake in fp16-safe form."""
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
        xt = n1(x, s, m)
        xt = _snake(xt, a1)
        xt = c1(xt)
        xt = n2(xt, s, m)
        xt = _snake(xt, a2)
        xt = c2(xt)
        x = xt + x
    return x


def rewrite_snakes(generator) -> int:
    import types
    n = 0
    for module in generator.modules():
        if type(module).__name__ == "AdaINResBlock1":
            module.forward = types.MethodType(_resblock_forward, module)
            n += 1
    return n


def _exact(before, after, x, what):
    with torch.no_grad():
        ref, got = before(x), after(x)
    diff = float((ref - got).abs().max())
    print(f"{what}: shapes {tuple(ref.shape)} vs {tuple(got.shape)}, max |diff| {diff:.2e}")
    assert ref.shape == got.shape and diff < 1e-3, what


NO_ISTFT = "--no-istft" in sys.argv
NO_MASK = "--no-mask" in sys.argv
import os
CUT = int(os.environ.get("KOKORO_CUT", "0"))


def _cut_forward(self, x_pre, ref_s, har, mask=None, mask_x10=None, mask_x60=None):
    """Upstream GeneratorFromHar.forward (fixed shapes), returning early at KOKORO_CUT:
    1 after the first upsample and noise add, 2 after the first resblock group, 3 after the
    second upsample and noise add, 4 after the second resblock group. A bisection aid."""
    from export_synth import wrappers
    s = ref_s[:, : wrappers.CoreMLExportConstants.VOICE_BASELINE_DIM]
    gen = self.generator
    # KOKORO_REFLECT=identity|zero swaps the reflection pad (diagnostic, not exact).
    reflect = os.environ.get("KOKORO_REFLECT")
    if reflect == "identity" and not isinstance(gen.reflection_pad, nn.Identity):
        gen.reflection_pad = nn.Identity()
    elif reflect == "zero" and not isinstance(gen.reflection_pad, nn.ConstantPad1d):
        gen.reflection_pad = nn.ConstantPad1d((1, 0), 0.0)
    x = x_pre
    cur_mask = mask
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        m_source = self._align_mask_to(cur_mask, x_source.shape[-1])
        x_source = gen.noise_res[i](x_source, s, m=m_source)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        tx, ts = x.size(2), x_source.size(2)
        if ts < tx:
            x_source = F.pad(x_source, (0, tx - ts))
        elif ts > tx:
            x_source = x_source[:, :, :tx]
        x = x + x_source
        if CUT == 1 + 2 * i:
            return x
        cur_mask = self._align_mask_to(cur_mask, x.shape[-1])
        xs = None
        for j in range(gen.num_kernels):
            r = gen.resblocks[i * gen.num_kernels + j](x, s, m=cur_mask)
            xs = r if xs is None else xs + r
        x = xs / gen.num_kernels
        if CUT == 2 + 2 * i:
            return x
    x = F.leaky_relu(x)
    x = gen.conv_post(x)
    spec = torch.exp(x[:, : gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, gen.post_n_fft // 2 + 1 :, :])
    return gen.stft.inverse(spec, phase)


if CUT:
    from export_synth import wrappers as _w
    _w.GeneratorFromHar.forward = _cut_forward
    print(f"ANE rewrite: the generator is cut at point {CUT}")


def _align_nearest(m, target_t):
    """Upstream's mask alignment (nearest lower index, a gather the Neural Engine leaves on the
    processor) as nearest-neighbour upsampling by the integer factor plus a replicate pad for
    the frames past cur_t * factor. The same numbers; the ANE runs upsample_nearest natively."""
    if m is None:
        return None
    target_t, cur_t = int(target_t), int(m.shape[-1])
    if cur_t == target_t:
        return m
    factor = target_t // cur_t
    up = F.interpolate(m, scale_factor=factor, mode="nearest")
    extra = target_t - up.shape[-1]
    return F.pad(up, (0, extra), mode="replicate") if extra > 0 else up


if not os.environ.get("KOKORO_KEEP_GATHER"):
    from export_synth import wrappers as _w2
    _upstream_align = _w2.GeneratorFromHar._align_mask_to
    torch.manual_seed(3)
    for cur, tgt in ((240, 2400), (2400, 14401), (48, 480)):
        probe = (torch.arange(cur) < cur * 0.7).float().reshape(1, 1, cur)
        a, b = _upstream_align(probe, tgt), _align_nearest(probe, tgt)
        assert a.shape == b.shape and torch.equal(a, b), ("mask alignment", cur, tgt)
    _w2.GeneratorFromHar._align_mask_to = staticmethod(_align_nearest)
    print("ANE rewrite: mask alignment by nearest upsampling (checked equal to upstream's gather)")

if os.environ.get("KOKORO_FP16_ALL"):
    import coremltools as _ct
    _convert = _ct.convert
    def _convert_all_fp16(*args, **kwargs):
        if kwargs.get("compute_precision") is not None:
            kwargs["compute_precision"] = _ct.transform.FP16ComputePrecision(op_selector=lambda op: True)
        return _convert(*args, **kwargs)
    _ct.convert = _convert_all_fp16
    cv.ct.convert = _convert_all_fp16
    print("ANE rewrite: every op at fp16 (no fp32 islands)")


def _unmask(generator):
    """Every AdaIN takes its statistics over the whole bucket, mask ignored, and the mask
    alignment (a gather per resolution) is gone. A diagnostic: it asks whether the mask
    arithmetic is what the Neural Engine compiler refuses. The app passes no mask today."""
    import types
    n = 0
    for module in generator.modules():
        if type(module).__name__ == "AdaIN1d":
            original = type(module).forward
            module.forward = types.MethodType(lambda self, x, s, m=None, _f=original: _f(self, x, s, None), module)
            n += 1
    from export_synth import wrappers
    wrappers.GeneratorFromHar._align_mask_to = staticmethod(lambda m, target_t: None)
    print(f"ANE rewrite: mask ignored in {n} AdaIN layers; mask alignment removed")

def _rewrite_both(generator):
    torch.manual_seed(2)
    ups = 0
    for i, up in enumerate(generator.ups):
        if isinstance(up, PolyphaseUpsample):
            continue
        new = PolyphaseUpsample(up)
        _exact(up, new, torch.randn(1, up.in_channels, 37), f"polyphase upsample {i}")
        generator.ups[i] = new
        ups += 1
    noise = 0
    for i, nc in enumerate(generator.noise_convs):
        if isinstance(nc, nn.Conv1d) and nc.stride[0] > 1:
            new = PolyphaseStrided(nc)
            _exact(nc, new, torch.randn(1, nc.in_channels, 14401), f"polyphase noise conv {i}")
            generator.noise_convs[i] = new
            noise += 1
    dil = rewrite_dilated(generator)
    if not os.environ.get("KOKORO_KEEP_SNAKE"):
        blocks = rewrite_snakes(generator)
        print(f"ANE rewrite: Snake in fp16-safe form in {blocks} resblocks")
    if not os.environ.get("KOKORO_KEEP_REFLECT"):
        pad = SliceReflectPad()
        _exact(generator.reflection_pad, pad, torch.randn(1, 128, 97), "slice reflect pad")
        generator.reflection_pad = pad
    if NO_MASK:
        _unmask(generator)
    print(f"ANE rewrite: {ups} upsample layers and {noise} strided noise convolutions in polyphase form, "
          f"{dil} dilated convolutions turned into phase convolutions")
    if NO_ISTFT:
        # The inverse STFT (20-tap kernels the ANE compiler refuses) leaves the
        # graph: the stage ends at the spectrum, [B, 11 magnitude + 11 phase, T],
        # and Swift finishes it with Accelerate, as it already makes the
        # harmonic source's forward STFT.
        # Flattened to one row: the converter checks the last axis against the
        # bucket's sample count. The consumer reads it back as [22, frames].
        # [B, 22, frames]: the magnitude rows then the phase rows. Not flattened: the
        # Neural Engine would see one axis 316,822 wide. The converter's waveform
        # gate is skipped for this shape (KOKORO_SKIP_WAVEFORM_GATE, a local patch).
        generator.stft.inverse = lambda spec, phase, length=None: torch.cat([spec, phase], dim=1)
        print("ANE rewrite: the inverse STFT is out of the graph; the stage ends at the spectrum")
    return ups
cv.rewrite_generator_ups_conv_transpose = _rewrite_both


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="3s")
    ap.add_argument("--out", default=str(HERE / "kokoro-coreml-ane"))
    ap.add_argument("--no-istft", action="store_true", help="end the generator at the spectrum")
    ap.add_argument("--no-mask", action="store_true", help="diagnostic: ignore the mask in every AdaIN")
    a = ap.parse_args()
    # A quick exactness check of the phase convolution on its own.
    torch.manual_seed(0)
    conv = nn.Conv1d(8, 8, 11, 1, dilation=5, padding=25)
    x = torch.randn(1, 8, 301)
    with torch.no_grad():
        ref = conv(x); got = PhaseConv1d(conv)(x)
    print("phase conv exactness, max |diff|:", float((ref - got).abs().max()))
    assert float((ref - got).abs().max()) < 1e-4
    cv.export_synthesizers(a.out, a.buckets, precision="float16", mode="decoder-har", rewrite_ups_conv_transpose=True)
