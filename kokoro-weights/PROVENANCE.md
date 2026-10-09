# Kokoro weights, held locally

John, 2026-10-09: "grab the files and use them here so we don't get things from unknown sources
repeatedly. You never know when the origin may be compromised." Everything the export and the
voice lab need is in this repo's folder; nothing is fetched at run time (`HF_HUB_OFFLINE=1`).

| file | origin | SHA-256 | checked |
|---|---|---|---|
| kokoro-v1_0.pth (327,212,226 bytes) | hexgrad/Kokoro-82M on Hugging Face, through the local cache the first export filled on 2026-10-04 | 496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4 | matches the hash Hugging Face publishes for the file (read once, 2026-10-09) |
| config.json (2,351 bytes) | same | 5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f | size matches; Hugging Face publishes no hash for small files |

`kokoro-coreml-export/checkpoints/` links here. The voice packs live in the app repo
(`text2Ear/Resources/KokoroVoices`, pinned SHA-256 in its PROVENANCE.md). The spaCy English model
misaki needs is in `../third-party/` with its hash. Verify at any time: `shasum -a 256 -c SHA256SUMS`.
