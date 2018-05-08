import collections
import itertools

def consume(iterator, n=None):
    if n is None:
        collections.deque(iterator, maxlen=0)
    else:
        next(itertools.islice(iterator, n, n), None)

def window(iterable, n=2):
    iterables = itertools.tee(iterable, n)
    for i, iterator in enumerate(iterables):
        consume(iterator, i)

    return zip(*iterables)
