#!/usr/bin/python3

import sys

import memm

def main(input_filename, feature_filename):
    extractor = memm.FeatureExtractor()
    extractor.update(open(input_filename, 'r').read())
    extractor.save(feature_filename)

if __name__ == '__main__':
    main(*sys.argv[1:])
