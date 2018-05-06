#!/usr/bin/python3

import pickle
import hmm
import patterns
import HMMTag
import GreedyTag


def main():
    counter = hmm.MLECounter(unknown_word_regexes=patterns.ENGLISH_PATTERNS)
    counter.update_by_q_summary(open('q.mle', 'r').read())
    counter.update_by_e_summary(open('e.mle', 'r').read())
    counter.w = pickle.load(open('extra', 'rb'))
    sentence = "I want to go to the mall"
    print(GreedyTag.greedy_tag_sentence(sentence, counter.get_score, counter.tagset))
    print(HMMTag.Viterbi(sentence, counter.get_score, counter.tagset, counter.possible_tags_per_word))

if __name__ == '__main__':
    main()
