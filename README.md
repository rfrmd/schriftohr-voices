# SchriftOhr Voices — Kokoro model mirror

A byte-faithful mirror of the [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml)
assets that SchriftOhr's neural narration downloads at first run: the ANE
CoreML bundles, the G2P models, the pronunciation lexicons, and the Kokoro
voice-pack sources. Served via GitHub Pages in the same URL shapes as the
Hugging Face Hub (`…/resolve/main/…` files and `…/api/models/…/tree/main`
listings), so the app can use this mirror as its primary registry with the
original repo as fallback.

Mirrored: `ANE/` (English model bundles + default voice), root lexicons and
G2P bundles, and `voices/` (voice-pack sources, kept for preservation).
Not mirrored: the retired mono-Kokoro bundles and the ANE-zh / ANE-ja
variants the app does not ship.

Upstream provenance: FluidInference's CoreML conversion of
[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
Apache-2.0. This mirror redistributes those files unmodified under the
same license. See the upstream repos for model cards and training details.
