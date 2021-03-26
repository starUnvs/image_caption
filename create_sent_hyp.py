'''
将模型hypothese从token变为文本
'''

import json

def untokenize(l):
    sentence=''
    for i, word in enumerate(l):
        if(i==0):
            sentence+=word
        else:
            sentence+=' '+word
    
    return sentence

with open('./ref_hyp.json', 'r') as j:
    ref_hyp = json.load(j)

# Load word map (word2ix)
with open('../preprocessed_data/WORDMAP_rsicd.json', 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

references = ref_hyp['references']
hypotheses = ref_hyp['hypotheses']

# index to raw sentence, every raw sentence is a list like ['i', 'love', 'you]
raw_references = list()
raw_hypotheses = list()

for refs in references:
    raw_refs = []
    for ref in refs:
        raw_refs.append([rev_word_map[index] for index in ref])
    
    raw_references.append(raw_refs)

for hyp in hypotheses:
    raw_hypotheses.append([rev_word_map[index] for index in hyp])


# make raw sentence to text(string type)
sent_references=[]
sent_hypotheses=[]
for raw_refs in raw_references:
    sent_references.append(list(map(untokenize, raw_refs)))

sent_hypotheses = list(map(untokenize, raw_hypotheses))


with open('ref_hyp_sentence.json', 'w') as j:
    json.dump({'references':sent_references, 'hypotheses': sent_hypotheses}, j)
