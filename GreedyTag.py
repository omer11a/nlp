import numpy as np
import sys
from collections import OrderedDict
import hmm
import pickle
import patterns
import operator

def greedy_tag_sentence(sentence, get_score, tagset):
    x = sentence.split()
    y = []
    for index, word in enumerate(x):
        prev_tag = y[index - 1] if index >= 1 else hmm.MLECounter.START_TAG
        prev_prev_tag = y[index - 2] if index >= 2 else hmm.MLECounter.START_TAG
        y.append(max([( tag, get_score(x,index, tag, prev_tag, prev_prev_tag) ) for tag in tagset], key=operator.itemgetter(1))[0])
        #print({tag : get_score(x, index, tag, prev_tag, prev_prev_tag) for tag in tagset})
    word_tag_tuple = zip(x,y)
    return ' '.join(('{0}/{1}'.format(word, tag) for (word, tag) in word_tag_tuple))

def main(input_file_name, q_mle_filename, e_mle_filename, out_file_name, extra_file_name):
    counter = hmm.MLECounter(unknown_word_regexes=patterns.ENGLISH_PATTERNS)
    counter.update_by_q_summary(open(q_mle_filename, 'r').read())
    counter.update_by_e_summary(open(e_mle_filename, 'r').read())
    counter.w = pickle.load(open(extra_file_name, 'rb'))

    tagged_document = (greedy_tag_sentence(line, counter.get_score, counter.tagset) for line in open(input_file_name, 'r').readlines())
    open(out_file_name, 'w').write('\n'.join(tagged_document))

if __name__ == '__main__':
     main(*sys.argv[1:])