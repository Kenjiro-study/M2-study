import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()

def detokenize(tokens):
    return detokenizer.detokenize(tokens)

def tokenize(utterance, lowercase=True):
    if lowercase:
        utterance = utterance.lower()
    tokens = word_tokenize(utterance)
    return tokens