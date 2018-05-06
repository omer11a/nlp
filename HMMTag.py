import pickle

import hmm
import patterns
import sys
from collections import namedtuple
import operator
import math

def GetWordTagset(word, tag_dict, tagset):
    if word in tag_dict:
        return tag_dict[word]
    return tagset

def Viterbi(sentence, get_score, tagset, tag_dict):
    Vkey = namedtuple("Vkey", ["prevtag", "tag"])
    words = sentence.split()
    ret = [None] * len(words)
    V = [{} for i in range(len(words))]
    bp = [{} for i in range(len(words))]

    for r in tagset:
        V[0][Vkey('START', r)] = myfix(0, get_score(words, 0, r))

    for r in tagset:
        for t in tagset:
            V[1][Vkey(t, r)] = myfix(V[0][Vkey('START', t)], get_score(words, 1, r, t))

    for i in range(2, len(words)):
        for t in GetWordTagset(words[i-1], tag_dict, tagset):
            for r in GetWordTagset(words[i], tag_dict, tagset):
                bp[i][Vkey(t, r)], V[i][Vkey(t, r)] = max([(tt,  myfix(V[i-1][Vkey(tt, t)], get_score(words, i, r, t, tt)))
                                                       for tt in GetWordTagset(words[i-2], tag_dict, tagset)], key = operator.itemgetter(1))

    ret[len(words) - 2], ret[len(words) - 1] = max([(k, v) for (k, v) in V[len(words)-1].items()],
                                                   key = operator.itemgetter(1))[0]

    for i in range(len(words) - 3, -1, -1):
        ret[i] = bp[i + 2][Vkey(ret[i + 1], ret[i + 2])]
    word_tag_tuple = zip(words,ret)
    return ' '.join(('{0}/{1}'.format(word, tag) for (word, tag) in word_tag_tuple))

def myfix(x,y):
    if x == -math.inf or y == 0:
        return -math.inf;
    else:
        return x+math.log(y)

def main(input_file_name, q_mle_filename, e_mle_filename, out_file_name, extra_file_name):
    counter = hmm.MLECounter(unknown_word_regexes=patterns.ENGLISH_PATTERNS)
    counter.update_by_q_summary(open(q_mle_filename, 'r').read())
    counter.update_by_e_summary(open(e_mle_filename, 'r').read())
    counter.w = pickle.load(open(extra_file_name, 'rb'))

    tagged_document = (Viterbi(line, counter.get_score, counter.tagset, counter.possible_tags_per_word) for line in open(input_file_name, 'r').readlines())
    open(out_file_name, 'w').write('\n'.join(tagged_document))

if __name__ == '__main__':
     main(*sys.argv[1:])