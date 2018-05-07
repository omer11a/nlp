import collections

import recipes

WORD_FEATURE = 'word'

def get_pos_paper_features(sequence, i):
    general_features = {}
    common_word_features = {}
    rare_word_features = {}

    words, tags = zip(*sequence)
    word = words[i]

    common_word_features[WORD_FEATURE] = word

    for n in range(1, min(4, len(word)) + 1):
        rare_word_features['prefix' + str(n)] = word[:n]
        rare_word_features['suffix' + str(n)] = word[-n:]

    rare_word_features['contains-number'] = any(char.isdigit() for char in word)
    rare_word_features['contains-upper'] = any(char.isupper() for char in word)
    rare_word_features['contains-hyphen'] = '-' in word

    general_features['prev-tag'] = tags[i - 1]
    general_features['prev-prev-tag'] = '%s %s' % (tags[i - 2], tags[i -1])

    general_features['prev-word'] = words[i - 1]
    general_features['prev-prev-word'] = words[i - 2]
    general_features['next-word'] = words[i + 1]
    general_features['next-next-word'] = words[i + 2]

    common_word_features.update(general_features)
    rare_word_features.update(general_features)

    return (common_word_features, rare_word_features)

class FeatureExtractor():
    START_TAG = 'START'
    END_TAG = 'END'
    DEFAULT_PREV_WINDOW_SIZE = 2
    DEFAULT_NEXT_WINDOW_SIZE = 2
    DEFAULT_THRESHOLD = 1

    def _update_features_by_phrase(self, tagged_phrase):
        start = (('', type(self).START_TAG), ) * self._prev_window_size
        end = (('', type(self).END_TAG), ) * self._next_window_size
        tagged_phrase_with_boundries = start + tagged_phrase + end
        for sequence in recipes.window(tagged_phrase_with_boundries, self._window_size):
            common_word_features, rare_word_features = type(self)._get_features(sequence, self._prev_window_size)
            self._common_word_examples.append(common_word_features)
            self._rare_word_examples.append(rare_word_features)

    def __init__(
        self,
        prev_window_size=DEFAULT_PREV_WINDOW_SIZE,
        next_window_size=DEFAULT_NEXT_WINDOW_SIZE,
        get_features=None
    ):
        self._prev_window_size = prev_window_size
        self._next_window_size = next_window_size
        self._window_size = prev_window_size + 1 + next_window_size
        self._get_features = get_pos_paper_features if get_features is None else get_features

        self._word_counter = collections.Counter()
        self._words = []
        self._tags = []
        self._common_word_examples = []
        self._rare_word_examples = []

    def update(self, tagged_text):
        for line in tagged_text.splitlines():
            tagged_phrase = tuple(tuple(token.rsplit('/', 1)) for token in line.split())
            words, tags = zip(*tagged_phrase)
            self._words += words
            self._tags += tags
            self._word_counter.update(words)
            self._update_features_by_phrase(tagged_phrase)

    def save(self, filename, threshold=DEFAULT_THRESHOLD):
        with open(filename, 'w') as output_file:
            for i, word in enumerate(self._words):
                features = self._common_word_examples[i]
                if self._word_counter[word] <= threshold:
                    features = self._rare_word_examples[i]

                feature_summary = ' '.join('{}={}'.format(name, value) for name, value in features.items())
                output_file.write('{} {}\n'.format(self._tags[i], feature_summary))
