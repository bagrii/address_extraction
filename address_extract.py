# Extract address from unstructured text

import pickle
import nltk
import string

from nltk import pos_tag
from nltk import word_tokenize
from nltk.chunk import ChunkParserI
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tag import ClassifierBasedTagger
from nltk.tag.util import untag
from nltk.stem.snowball import SnowballStemmer

# IOB tag name for specifying address 
GPE_TAG = "GPE"

class AddressChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=self.features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)
    
    def features(self, tokens, index, history):
        # for more details see: http://nlpforhackers.io/named-entity-extraction/ 
        
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """

        # init the stemmer
        stemmer = SnowballStemmer('english')

        # Pad the sequence with placeholders
        tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
        history = ['[START2]', '[START1]'] + list(history)

        # shift the index with 2, to accommodate the padding
        index += 2

        word, pos = tokens[index]
        prevword, prevpos = tokens[index - 1]
        prevprevword, prevprevpos = tokens[index - 2]
        nextword, nextpos = tokens[index + 1]
        nextnextword, nextnextpos = tokens[index + 2]
        previob = history[index - 1]
        contains_dash = '-' in word
        contains_dot = '.' in word
        allascii = all([True for c in word if c in string.ascii_lowercase])

        allcaps = word == word.capitalize()
        capitalized = word[0] in string.ascii_uppercase

        prevallcaps = prevword == prevword.capitalize()
        prevcapitalized = prevword[0] in string.ascii_uppercase

        nextallcaps = prevword == prevword.capitalize()
        nextcapitalized = prevword[0] in string.ascii_uppercase

        f = {
            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': allascii,

            'next-word': nextword,
            'next-lemma': stemmer.stem(nextword),
            'next-pos': nextpos,

            'next-next-word': nextnextword,
            'nextnextpos': nextnextpos,

            'prev-word': prevword,
            'prev-lemma': stemmer.stem(prevword),
            'prev-pos': prevpos,

            'prev-prev-word': prevprevword,
            'prev-prev-pos': prevprevpos,

            'prev-iob': previob,

            'contains-dash': contains_dash,
            'contains-dot': contains_dot,

            'all-caps': allcaps,
            'capitalized': capitalized,

            'prev-all-caps': prevallcaps,
            'prev-capitalized': prevcapitalized,

            'next-all-caps': nextallcaps,
            'next-capitalized': nextcapitalized,
        }

        return f

def get_address_chunker(dataset_file_name):
    """
    returns AddressChunker instance with dataset_file_name as training samples

    `dataset_file_name` = file name of pickled list of CoNLL IOB format sentences
    """

    with open(dataset_file_name, 'rb') as fp:
        dataset = pickle.load(fp)
        print(len(dataset))
        chunker = AddressChunker(dataset)

    return chunker

def get_chuncker_accuracy(chunker, test_samples):
    """
    returns score of the chunker against the gold standard
    """
    score = chunker.evaluate([
        conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
        for iobs in test_samples
        ])
    return score.accuracy()

def get_tagged_sentence(chunker, sentence):
    """
    returns IOB tagged tree of sentence
    """
    return chunker.parse(pos_tag(word_tokenize(sentence)))

def extract_address(chunker, sentence):
    """
    returns all addresses in sentence
    """
    def tree_filter(tree):
        return GPE_TAG == tree.label()

    tagged_tree = get_tagged_sentence(chunker, sentence)
    addresses = list()
    for subtree in tagged_tree.subtrees(filter=tree_filter):
        addresses.append(untag(subtree.leaves()))
    return addresses

print("Loading dataset...")
chunker = get_address_chunker('dataset/IOB_tagged_addresses.pkl')
print("Done.")
print(extract_address(chunker, "Hey man! Joe lives here: 44 West 22nd Street, New York, NY 12345. Can you contact him now? If you need any help, call me on 12345678"))
