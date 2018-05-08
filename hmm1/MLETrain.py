#!/usr/bin/python3

import sys
import pickle

import hmm
import patterns

def main(input_filename, q_mle_filename, e_mle_filename, extra_filename=None):
    counter = hmm.MLECounter(unknown_word_regexes=patterns.ENGLISH_PATTERNS)
    counter.update(open(input_filename, 'r').read())
    counter.update_uncommon(1)

    open(q_mle_filename, 'w').write(counter.get_q_summary())
    open(e_mle_filename, 'w').write(counter.get_e_summary())

    if extra_filename is not None:
        counter.learn_weights()
        pickle.dump(counter.w, open(extra_filename, 'wb'))

if __name__ == '__main__':
    main(*sys.argv[1:])
