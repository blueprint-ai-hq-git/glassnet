# Phase 6 switch demo — the experimental trainer behind the switch.
# The normal SGD auto-pilot stays the default; this shows the alternative.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassnet.api import GlassNet

model = GlassNet("fashion-mnist")
model.train(optimizer="evo")          # experimental: (1+1)-ES, no gradients
print("evo test accuracy: %.2f%%" % (model.accuracy() * 100))
print(model.explain(18))