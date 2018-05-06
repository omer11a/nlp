import collections
import re

import recipes

class MLECounter:
    START_TAG = 'START'
    SIGNATURE_PREFIX = '^'
    DEFAULT_SIGNATURE = '^unknown'
    DEFAULT_THRESHOLD = 1

    def _update_tag_sequence_counter(self, tags):
        for n in range(1, 4):
            tags_with_start_tokens = (type(self).START_TAG, ) * n + tags
            sequences = recipes.window(tags_with_start_tokens, n)
            self._tag_sequence_counter.update(sequences)

    def _match_unknown_word(self, word):
        for signature, pattern in self._unknown_word_patterns.items():
            if pattern.fullmatch(word) is not None:
                return signature

        return type(self).DEFAULT_SIGNATURE

    def __get_e_count_summary(self, word_tag_counter):
        lines = []
        for word, counter in word_tag_counter.items():
            lines += ["%s %s\t%d" % (word, tag, count) for tag, count in counter.items()]

        return '\n'.join(lines)

    def _get_tag_sequence_count(self, tags):
        if len(tags) == 0:
            return sum(self._word_counter.values())

        return self._tag_sequence_counter[tuple(tags)]

    def _get_tag_sequence_probability(self, tags, should_ignore_one=False):
        numerator = self._get_tag_sequence_count(tags)
        if should_ignore_one:
            numerator -= 1

        if numerator <= 0:
            return 0

        denominator = self._get_tag_sequence_count(tags[:-1])
        if should_ignore_one:
            denominator -= 1

        return numerator / denominator

    def _get_q(self, tag, prev_tag, prev_prev_tag):
        p1 = self._get_tag_sequence_probability((prev_prev_tag, prev_tag, tag))
        p2 = self._get_tag_sequence_probability((prev_tag, tag))
        p3 = self._get_tag_sequence_probability((tag, ))

        return self._w[0] * p1 + self._w[1] * p2 + self._w[2] * p3

    def _get_e(self, word, tag):
        denominator = self._get_tag_sequence_count((tag, ))
        if denominator <= 0:
            return 0

        if self._word_counter[word] > 0:
            numerator = self._word_tag_counter[word][tag]
        else:
            signature = self._match_unknown_word(word)
            numerator = self._unknown_word_tag_counter[signature][tag]

        return numerator / denominator

    def _get_score(self, word, tag, prev_tag=START_TAG, prev_prev_tag=START_TAG):
        return self._get_q(tag, prev_tag, prev_prev_tag) * self._get_e(word, tag)

    def __init__(self, unknown_word_regexes=None, w=None):
        self._tagset = set()

        self._word_counter = collections.Counter()
        self._tag_sequence_counter = collections.Counter()
        self._word_tag_counter = collections.defaultdict(collections.Counter)
        self._unknown_word_tag_counter = collections.defaultdict(collections.Counter)
        self._possible_tags_per_word = collections.defaultdict(set)

        unknown_word_regexes = {} if unknown_word_regexes is None else unknown_word_regexes
        self._unknown_word_patterns = {
            signature : re.compile(regex)
            for signature, regex in unknown_word_regexes.items()
        }

        self._w = (1/3, 1/3, 1/3) if w is None else w

    @property
    def tagset(self):
        return self._tagset

    @property
    def possible_tags_per_word(self):
        return self._possible_tags_per_word

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = tuple(value)

    def update(self, tagged_text):
        for line in tagged_text.splitlines():
            tagged_phrase = [tuple(token.rsplit('/', 1)) for token in line.split()]
            words, tags = zip(*tagged_phrase)

            self._tagset.update(tags)
            self._word_counter.update(words)
            self._update_tag_sequence_counter(tags)

            for word, tag in tagged_phrase:
                self._word_tag_counter[word].update((tag, ))
                self._possible_tags_per_word[word].add(tag)

    def update_uncommon(self, threshold=DEFAULT_THRESHOLD):
        uncommon_words = [word for word, count in self._word_counter.items() if count <= threshold]
        for word in uncommon_words:
            signature = self._match_unknown_word(word)
            self._unknown_word_tag_counter[signature].update(self._word_tag_counter[word])
            del self._word_tag_counter[word]
            del self._possible_tags_per_word[word]

    def get_q_summary(self):
        summary = ("%s\t%d" % (' '.join(tags), count) for tags, count in self._tag_sequence_counter.items())
        return '\n'.join(summary)

    def get_e_summary(self):
        word_summary = self.__get_e_count_summary(self._word_tag_counter)
        unknown_word_summary = self.__get_e_count_summary(self._unknown_word_tag_counter)
        return '\n'.join((word_summary, unknown_word_summary))

    def update_by_q_summary(self, summary):
        for line in summary.splitlines():
            tags, count = line.split('\t')
            self._tag_sequence_counter[tuple(tags.split())] += int(count)

    def update_by_e_summary(self, summary):
        for line in summary.splitlines():
            word, tag, count = line.split()

            count = int(count)
            self._tagset.add(tag)
            if word.startswith(type(self).SIGNATURE_PREFIX):
                self._word_counter[type(self).DEFAULT_SIGNATURE] += count
                self._unknown_word_tag_counter[word][tag] += count
            else:
                self._word_counter[word] += count
                self._word_tag_counter[word][tag] += count
                self._possible_tags_per_word[word].add(tag)

    def learn_weights(self):
        w = [0, 0, 0]

        trigrams = [tags for tags in self._tag_sequence_counter if len(tags) == 3]
        for trigram in trigrams:
            p1 = self._get_tag_sequence_probability(trigram, should_ignore_one=True)
            p2 = self._get_tag_sequence_probability(trigram[1:], should_ignore_one=True)
            p3 = self._get_tag_sequence_probability(trigram[2:], should_ignore_one=True)

            p = (p1, p2, p3)
            w[p.index(max(p))] += self._get_tag_sequence_count(trigram)

        weight_sum = sum(w)
        self._w = (w[0] / weight_sum, w[1] / weight_sum, w[2] / weight_sum)

    def get_score(self, words, i, tag, prev_tag = START_TAG, prev_prev_tag = START_TAG):
        return self._get_score(words[i], tag, prev_tag, prev_prev_tag)
