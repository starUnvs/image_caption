import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Parameters
# folder with data files saved by create_input_files.py
data_folder = '../preprocessed_data'
data_name = 'rsicd_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# model checkpoint
checkpoint = './BEST_checkpoint_rsicd_new.pth.tar'
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = '../preprocessed_data/WORDMAP_rsicd.json'
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST'),
        batch_size=1, shuffle=True, pin_memory=True)

    # Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        info, relation = encoder(image)
        info_dim = info.shape[3]

        # Flatten encoding
        info = info.view(1, -1, info_dim)
        relation = relation.view(1, -1)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        info = info.expand(k, info.shape[1], info.shape[2])
        relation = relation.expand(k, relation.shape[1])

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c, C = decoder._init_hidden_state(info, relation)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            scores, (h,c,C), _ = decoder.next_pred(embeddings, info, (h, c, C))
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            C = C[prev_word_inds[incomplete_inds]]
            relation = relation[prev_word_inds[incomplete_inds]]
            info = info[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        text = [rev_word_map[s] for s in seq]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {
                          word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    with open('./ref_hyp.json', 'w') as j:
        json.dump({'references': references, 'hypotheses': hypotheses}, j)


def list_to_sentence(l):
    sentence = ''
    for i, word in enumerate(l):
        if(i == 0):
            sentence += word
        else:
            sentence += ' '+word

    return sentence


if __name__ == '__main__':
    beam_size = 5
    evaluate(beam_size)

    with open('./ref_hyp.json', 'r') as j:
        ref_hyp = json.load(j)

    # Load word map (word2ix)
    with open('../preprocessed_data/WORDMAP_rsicd.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    references = ref_hyp['references']
    hypotheses = ref_hyp['hypotheses']

    raw_references = list()
    raw_hypotheses = list()

    for refs in references:
        raw_refs = []
        for ref in refs:
            raw_refs.append([rev_word_map[index] for index in ref])

        raw_references.append(raw_refs)

    for hyp in hypotheses:
        raw_hypotheses.append([rev_word_map[index] for index in hyp])
    sent_references = []
    sent_hypotheses = []
    for raw_refs in raw_references:
        sent_references.append(list(map(list_to_sentence, raw_refs)))

    sent_hypotheses = list(map(list_to_sentence, raw_hypotheses))

    with open('ref_hyp_sentence.json', 'w') as j:
        json.dump({'references': sent_references,
                   'hypotheses': sent_hypotheses}, j)
