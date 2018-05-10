#!/usr/bin/python3

import sys
import functools

import memm
import functools

def main(input_filename, feature_filename):
    get_ner_features = functools.partial(memm.get_ner_features, memm.load_lexicons())
    extractor = memm.FeatureExtractor(get_features=get_ner_features)
    extractor.update(open(input_filename, 'r').read())
    extractor.save(feature_filename)

if __name__ == '__main__':
    main(*sys.argv[1:])
