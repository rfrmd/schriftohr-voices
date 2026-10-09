#!/usr/bin/env python3
"""Does a clause cut show? One long sentence (past the 10 s bucket) rendered whole, and as two
clause pieces with the pause its semicolon carries, for Heart and for John. John, 2026-10-09:
"have you found a way to make it imperceptible to the listener?"
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("voice_lab", HERE / "voice-lab.py")
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)

WHOLE = ("When at last the travellers came down out of the hills into the valley, where the river ran broad "
         "and slow between the willows, they found the town already asleep; not a light showed in any window, "
         "and the only sound was the water moving under the old stone bridge.")
PIECES = [("When at last the travellers came down out of the hills into the valley, where the river ran broad "
           "and slow between the willows, they found the town already asleep;", 0.22),
          ("not a light showed in any window, and the only sound was the water moving under the old stone bridge.", 0.0)]

out = Path(sys.argv[1]).expanduser(); out.mkdir(parents=True, exist_ok=True)
L = lab.Lab()
heart = lab.pack("af_heart")
males = [lab.pack(n) for n in ["am_adam", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]]
females = [heart] + [lab.pack(n) for n in ["af_bella", "af_kore", "af_nicole", "af_nova", "af_sarah", "bf_emma", "bf_isabella"]]
john = heart + 1.6 * (np.mean(males, axis=0) - np.mean(females, axis=0))
rows = ["| file | how | length s |", "|---|---|---|"]
for vname, voice in (("heart", heart), ("john", john)):
    lab.PASSAGE = WHOLE
    whole = L.render(voice)
    parts = []
    for text, pause in PIECES:
        lab.PASSAGE = text
        parts.append(L.render(voice)); parts.append(np.zeros(int(pause * 24000), dtype=np.float32))
    cut = np.concatenate(parts)
    lab.write_wav(out / f"{vname}-whole.wav", whole); lab.write_wav(out / f"{vname}-cut-at-semicolon.wav", cut)
    rows += [f"| {vname}-whole.wav | one piece, as a 15 s bucket would render it | {len(whole)/24000:.1f} |",
             f"| {vname}-cut-at-semicolon.wav | two pieces, 0.22 s pause at the semicolon, as the 10 s cap would render it | {len(cut)/24000:.1f} |"]
    print("CUT", vname, f"whole {len(whole)/24000:.1f} s, cut {len(cut)/24000:.1f} s")
(out / "README.md").write_text("# One long sentence, whole and cut at its semicolon, 2026-10-09\n\n" + WHOLE + "\n\n" + "\n".join(rows) + "\n")
