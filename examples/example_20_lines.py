# Phase 5 GATE — the whole library in (fewer than) 20 lines.
# Trains, explains, reports, and scores data — start to finish.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassnet.api import GlassNet

model = GlassNet("fashion-mnist")          # 1. make the model
model.train()                               # 2. train (auto-pilot, zero settings)
print("test accuracy: %.2f%%" % (model.accuracy() * 100))
print(model.explain(18, save="docs/assets/reports/report_18.png"))   # 3. why? text + picture
print(model.explain(42, save="docs/assets/reports/report_42.png"))
model.train_judge()                         # 4. the data judge
print("judge scores (first 5 test images):",
      model.judge(model.X_test[:5]).round(2).tolist())