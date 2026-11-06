import json, sys
p = 'data/chat/prepared_mix_ctrl_honest/vocab_v12000.json'
v = json.load(open(p))
print('type', type(v), 'len', len(v))
if isinstance(v, dict):
    items = list(v.items())[:25]
    for k, val in items:
        print(repr(k), '->', repr(val))
else:
    for t in v[:25]:
        print(repr(t))
print('--- meta ---')
print(open('data/chat/prepared_mix_ctrl_honest/meta_v12000.json').read()[:1200])
