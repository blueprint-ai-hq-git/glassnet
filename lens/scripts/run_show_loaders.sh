#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u -c "
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
from lens.vision import load_vision

def show(f):
    try:
        net = load_vision(f)
        print(os.path.basename(f), '->', type(net).__name__, end=' ')
        print('img_concepts', len(getattr(net, 'img_concepts', []) or []),
              'txt', len(getattr(net, 'txt_concepts', []) or []), end=' ')
        try:
            meta = json.loads(str(np.load(f, allow_pickle=False)['__meta__']))
            print('metaV', meta.get('V'), 'avocab', (meta.get('avocab') or [])[:4],
                  'qvocab', (meta.get('qvocab') or [])[:3])
        except Exception as e:
            print(' meta read err', repr(e)[:50])
    except Exception as e:
        print(os.path.basename(f), 'FAIL', repr(e)[:60])

for f in sorted(os.listdir('lens/runs/image')):
    if f.endswith('.npz'):
        show('lens/runs/image/' + f)
for f in sorted(os.listdir('lens/runs/vl')):
    if f.endswith('.npz'):
        show('lens/runs/vl/' + f)
" 2>&1 | tail -25
