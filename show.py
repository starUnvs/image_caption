import json
import sys

with open('ref_hyp_sentence.json', 'r') as j:
    ref_hyp = json.load(j)

f = open('./results.txt', 'w')

references = ref_hyp['references']
hypotheses = ref_hyp['hypotheses']

for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
    print('%d--[references]:'%(i), file=f)
    for i, r in enumerate(ref):
        print('%d. %s' % (i, r), file=f)
    print('[hypotheses]', file=f)
    print(hyp, file=f)
    print('', file=f)
