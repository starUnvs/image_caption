# 根据ref_hyp.json与ref_hyp_sentence.json计算bleu,meteor,rouge得分
# bleu与meteor得分由nltk库计算，rouge得分由rouge package计算，需通过pip下载 

import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
from rouge import Rouge

bleu1_weight = [1.]
bleu2_weight = [1./2., 1./2.]
bleu3_weight = [1./3., 1./3., 1./3.]
bleu4_weight = [0.25, 0.25, 0.25, 0.25]

bleu_weights = [bleu1_weight, bleu2_weight, bleu3_weight, bleu4_weight]

with open('./ref_hyp.json') as j:
    ref_hyp = json.load(j)

with open('./ref_hyp_sentence.json') as j:
    ref_hyp_sentence = json.load(j)

bleu_scores = [corpus_bleu(ref_hyp['references'], ref_hyp['hypotheses'], weights=w) for w in bleu_weights]
print(bleu_scores)


meteor_scores = []
for ref, hyp in zip(ref_hyp_sentence['references'], ref_hyp_sentence['hypotheses']):
    meteor_scores.append(meteor_score(ref, hyp))
print(np.mean(meteor_scores))


rouge = Rouge()
hyp_index = []
for ref, hyp in zip(ref_hyp_sentence['references'], ref_hyp_sentence['hypotheses']):
    scores = [rouge.get_scores(r, hyp)[0]['rouge-l']['f'] for r in ref]
    hyp_index.append(np.argmax(scores))

best_ref_sentences = [refs[index] for refs, index in zip(
    ref_hyp_sentence['references'], hyp_index)]
rougeL_score = rouge.get_scores(
    best_ref_sentences, ref_hyp_sentence['hypotheses'], avg=True)
print(rougeL_score)
