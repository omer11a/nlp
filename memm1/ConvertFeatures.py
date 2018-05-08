#!/usr/bin/python3

import sys
import re
import collections
import sklearn.feature_extraction

import memm

def main(input_filename, feature_vector_filename, feature_map_filename):
    tags = []
    tag_to_index = {}
    examples = []

    with open(input_filename, 'r') as input_file:
        for line in input_file:
            tag, rest_of_line = line.split(' ', 1)
            tags.append(tag)
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)

            feature_descriptions = re.findall(r'[\w-]+=[^=]*(?=\s)', rest_of_line)
            features = dict(description.split('=') for description in feature_descriptions)
            examples.append(features)

    possible_tags_per_word = collections.defaultdict(set)
    for i, example in enumerate(examples):
        if memm.WORD_FEATURE in example:
            word = example[memm.WORD_FEATURE]
            possible_tags_per_word[word].add(tags[i])

    vectorizer = sklearn.feature_extraction.DictVectorizer()
    matrix = vectorizer.fit_transform(examples)

    with open(feature_vector_filename, 'w') as feature_vector_file:
        for i, vector in enumerate(matrix):
            tag_index = tag_to_index[tags[i]]
            nonzeros = vector.nonzero()[1]
            vector_as_text = ' '.join('%d:1' % (i, ) for i in nonzeros)
            feature_vector_file.write('%d %s\n' % (tag_index, vector_as_text))

    with open(feature_map_filename, 'w') as feature_map_file:
        for tag, index in tag_to_index.items():
            feature_map_file.write('%s %d\n' % (tag, index))

        for feature, index in vectorizer.vocabulary_.items():
            feature_map_file.write('%s %d\n' % (feature, index))

        for word, tags in possible_tags_per_word.items():
            feature_map_file.write('%s/%s\n' % (word, ' '.join(tags)))

if __name__ == '__main__':
    main(*sys.argv[1:])
