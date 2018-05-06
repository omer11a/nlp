import sys

def main(tagged_filename, our_tagged_filename):
    tagged_list = []
    our_tagged_list = []
    count_words = 0
    count_good_words = 0
    tagged_text = open(tagged_filename, 'r').readlines()
    our_tagged_text = open(our_tagged_filename, 'r').readlines()
    for line in tagged_text:
        tagged_phrase = [tuple(token.rsplit('/', 1)) for token in line.split()]
        words, tags = zip(*tagged_phrase)
        tagged_list.extend([tag for tag in tags])
    for line in our_tagged_text:
        tagged_phrase = [tuple(token.rsplit('/', 1)) for token in line.split()]
        words, tags = zip(*tagged_phrase)
        our_tagged_list.extend([tag for tag in tags])

    tags_tuples = zip(tagged_list, our_tagged_list)

    for (a, b) in tags_tuples:
        if a == b:
            count_good_words += 1
        count_words += 1

    print("Our Greedy accuracy is:{0}".format((count_good_words / count_words) *100))

if __name__ == '__main__':
     main(*sys.argv[1:])