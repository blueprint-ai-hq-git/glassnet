#!/usr/bin/env python3
"""lens/scripts/train_chat_sent.py -- launch the REAL trainer with the RUNNING SENTENCE STATE grafted on.

This file exists for ONE reason and it is declared, not hidden: the shipped trainer
(lens/scripts/train_chat.py) must not be edited while an arm is live (agree_sgd04_s2 has held it since
16:38 today), so the lever is installed from a separate module instead of being written into the trainer.

    SENTSTATE=0 (or unset)  -> the graft is installed but DISABLED: the epoch line must be field-identical
                               to the unmodified trainer's. That is P114a's control.
    SENTSTATE=1             -> the lever is ON: answer layer reads the window's concepts AND the carried
                               sentence state (lens/sentstate_wire.py).

Everything else -- argparse, the corpus loader, the epoch loop, the checkpoints, the gates -- is the shipped
trainer's own code, executed unmodified through runpy, so an arm launched here is the same experiment with one
lever added and nothing else re-implemented.

Predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt (written before this file existed).
"""
import os
import runpy
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from lens import sentstate_wire as W                                   # noqa: E402

DS = int(os.environ.get("SENTSTATE_DS", "64"))
W.install(ds=DS)
if os.environ.get("SENTSTATE", "0") == "1":
    W.enable(ds=DS)
    print(f"[train_chat_sent] SENTSTATE ON, ds={DS}: answer layer = concepts + carried sentence state "
          f"(+{W.ship_config_cost()[1]} params at the ship config, +13.47% on 5,908,686)", flush=True)
else:
    W.disable()
    print("[train_chat_sent] SENTSTATE OFF: the graft is installed but inert (control run)", flush=True)

runpy.run_path(os.path.join(PROJ, "lens", "scripts", "train_chat.py"), run_name="__main__")
