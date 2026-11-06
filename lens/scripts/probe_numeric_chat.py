"""Probe: does the REAL chat corpus contain numeric content, and where?
Reads only. Prints counts. No writes to data/."""
import json, numpy as np, random, re

D = 'data/chat/prepared_mix_ctrl_honest'
vocab = json.load(open(D + '/vocab_v12000.json'))
ids = np.load(D + '/ids_v12000.npy', mmap_mode='r')
bounds = np.load(D + '/bounds_v12000.npy')
split = np.load(D + '/split_v12000.npy')
print('ids', ids.shape, ids.dtype)
print('bounds', bounds.shape, bounds.dtype, 'first 8', bounds[:8].tolist())
print('split', split.shape, split.dtype, 'counts', np.bincount(split.astype(np.int64)).tolist())

NUM = re.compile(r'^[+-]?\d+(?:[.,]\d+)*$')
numtok = np.array([bool(NUM.match(t)) for t in vocab])
print('vocab numeric tokens:', int(numtok.sum()))
print('examples:', [t for t in vocab if NUM.match(t)][:30])

rng = random.Random(20261020)
n = int(ids.shape[0]) if ids.ndim == 1 else int(ids.shape[0])
W = 32
starts = [rng.randrange(0, n - W) for _ in range(2000)]
win = np.stack([np.asarray(ids[s:s + W]) for s in starts])
hasnum = numtok[win].sum(axis=1)
print('windows sampled:', len(starts))
print('windows with >=1 numeric token:', int((hasnum > 0).sum()), '=', round(float((hasnum > 0).mean()) * 100, 2), '%')
print('mean numeric tokens per window: %.3f' % float(hasnum.mean()))
for i in range(5):
    if hasnum[i] > 0:
        print('  EX', ' '.join(vocab[t] for t in win[i]))
# assistant turns: find the marker token
try:
    a = vocab.index('assistant')
    print('assistant token id', a, 'occurrences in sampled windows', int((win == a).sum()))
except ValueError:
    print('no literal assistant token')
print('first 60 ids:', np.asarray(ids[:60]).tolist())
print('first 60 tokens:', ' '.join(vocab[t] for t in np.asarray(ids[:60])))
