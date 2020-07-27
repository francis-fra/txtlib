from sklearn import metrics
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from txtlib import preprocessor as pp

import copy
from scipy.stats import itemfreq

import re
from collections import Counter
import re, string, random, glob, operator, heapq
from collections import defaultdict
from math import log10
from functools import reduce
import operator
import collections

from functools import partial
import json

import function_pipe as fpn

# VOCABULARY_DIR = '/home/fra/DataMart/datacentre/text_data/'
VOCABULARY_DIR = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\text_mining\\corpus\\data\\'
VOCABULARY_DATAFILE = VOCABULARY_DIR + 'big.txt'

# locations
# dataDir = '/home/fra/DataMart/datacentre/text_data/'
dataDir = VOCABULARY_DIR
ONEGRAMFILE = dataDir + 'count_1w.txt'
BIGRAMFILE = dataDir + 'count_2w.txt'


class SpellChecker(object):
    
    def __init__(self):
        "init"
        pass
        
    def read_vocabulary(self, vocabDataSrc=VOCABULARY_DATAFILE, upper=False):
        "set vocabulary list from file"
        self.words = Counter(word_tokenize(open(vocabDataSrc).read()))
        self.numWords = sum(self.words.values())
        
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        if upper:
            self.letters = self.letters.upper()
        else:
            self.letters = self.letters.lower()
        
    def set_vocabulary(self, vocabList, upper=False):
        "set vocabulary list"
        self.words = Counter(word_tokenize(vocabList))
        self.numWords = sum(self.words.values())
        
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        if upper:
            self.letters = self.letters.upper()
        else:
            self.letters = self.letters.lower()
        
    def P(self, wd, N=None):
        "probability of word"
        if N is None:
            N = self.numWords
        return (self.words[wd] / N)

    def known(self, words): 
        '''
            The subset of words that appear in the dictionary of WORDS.
            filter the words list which are known words
            :param:     words:   list or set of strings
            :return:             filter list of words which are in the vocabulary
        '''
        return set(w for w in words if w in self.words)


    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)
    
    def edits1(self, word):
        '''
            All edits that are one edit away from the given word
            :param:     word: a string
            :return:         set of string
        '''
        letters    = self.letters
        # list of single splits
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        # list of words with one letter removed
        deletes    = [L + R[1:]               for L, R in splits if R]
        # list of words with 2 letters swapped
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        # list of words with one letter replaced
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        # list of words with one letter inserted
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)    
        
    def edits2(self, word): 
        '''
            All edits that are two edits away from `word`
            :param: a string
            :return: set of string        
        '''
        return set([e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)])
    
    
#     def edits3(self, word): 
#         '''
#             All edits that are 3 edits away from `word`
#             :param: a string
#             :return: set of string        
#         '''
#         return set([e3 for e1 in self.edits1(word) for e2 in self.edits1(e1) for e3 in self.edits1(e2)])        
        
    def extend(self, prefix):
        "find words give the prefix"
        N = len(prefix)
        return set(words for words in self.words if words[:N] == prefix)
        
def get_empty_index(txt_list):
    return [idx for idx, doc in zip(range(len(txt_list)), txt_list) if not doc.strip()]
    
def filter_empty_txt(txt_list, indices):
    rng = range(len(txt_list)) 
    idx = list(set(rng) - set(indices))    
    return list( txt_list[k] for k in idx )

def memo(f):
    "Memorize function f"
    # store the function argument and output in a dictionary
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo


class WordSegment(object):
    
    def __init__(self):
        self.N = 1024908267229 
        self.setup()
    
    
    def setup(self, one_gram_file=ONEGRAMFILE, bi_gram_file=BIGRAMFILE):
        # 1-gram data dictionary: tuple (word, freq)
        self.Pw  = Pdist(self._datafile(one_gram_file), self.N, self.avoid_long_words)
        # bigram data dictionary: tuple (word, freq)
        self.P2w = Pdist(self._datafile(bi_gram_file), self.N)
        
        #print(one_gram_file)
        #print(bi_gram_file)
        
        
    # read file generator        
    def _datafile(self, name, sep='\t'):
        "Read (key, value) pairs from file."
        with open(name, 'r') as f:
            for line in f:
                yield line.split(sep)
                
            f.close()        
        
    @memo
    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        candidates = ([first]+self.segment(rem) for first,rem in self.splits(text))
        return max(candidates, key=Pwords)

    def splits(self, text, L=20):
        "Return a list of all possible (first, rem) pairs"
        # len(first) <= L
        return [(text[:i+1], text[i+1:]) for i in range(min(len(text), L))]
    
    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        # product of the probability of each word in the list
        product = lambda nums: reduce(operator.mul, nums, 1)
        return product(self.Pw(w) for w in words)
    
    def avoid_long_words(self, key, N):
        "Estimate the probability of an unknown word"
        return 10./(N * 10**len(key))

    def cPw(self, word, prev):
        "Conditional probability of word, given previous word - P(word | prev_word)"
        try:
            # P2(x + y) / P(x)
            return self.P2w[prev + ' ' + word]/float(self.Pw[prev])
        except KeyError:
            return Pw(word)


    @memo 
    def segment2(self, text, prev='<S>', all=False): 
        "Return (log P(words), words), where words is the best segmentation."
        
        def combine(self, Pfirst, first, tup):
            '''
                Add log probilities first and rem results into one (probability, word tokens) pair
                Pfirst:   probability of (prev + first)
                first:    first split word
                tup:      (prob, list of word tokens)
            '''
            Prem = tup[0]       # probability
            rem = tup[1]        # list of word tokens
            # add log probability
            return Pfirst+Prem, [first]+rem 
    
    
        if not text: return (0.0, [])
        # log prob(prev, first) + log prob(first, rem)
        candidates = [combine(log10(self.cPw(first, prev)), first, self.segment2(rem, first)) 
                      for first, rem in self.splits(text)]
        # return all for debug
        if all:
            return candidates
        else:
            return max(candidates)

    # TODO: find the most likely word given the previous
    def most_likely(self, prev):
        "the most likely word following prev"
        pass 



class Pdist(dict):
    'A probability distribution estimated from counts in datafile'
    
    def __init__(self, data=[], N=None, missingfn=None):
        '''
           :param: data: list of tuples
           :param: N:    number of tokens    
           :param: missingfn
        '''
        # N if N is not None, sum(...) otherwise            
        self.N = float(N or sum(self.itervalues()))        
        # filter function
        self.missingfn = missingfn or (lambda k, N: 1./N)
        # loop the tuple (word, count)
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        
    def __call__(self, key):
        "return value of dictionary"
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)




def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)
        

# def get_metrics(true_labels, predicted_labels):
#     "classification statistics"
#     
#     print ('Accuracy:', np.round(
#                         metrics.accuracy_score(true_labels, 
#                                                predicted_labels),
#                         2))
#     print ('Precision:', np.round(
#                         metrics.precision_score(true_labels, 
#                                                predicted_labels,
#                                                average='weighted'),
#                         2))
#     print ('Recall:', np.round(
#                         metrics.recall_score(true_labels, 
#                                                predicted_labels,
#                                                average='weighted'),
#                         2))
#     print ('F1 Score:', np.round(
#                         metrics.f1_score(true_labels, 
#                                                predicted_labels,
#                                                average='weighted'),
#                         2))
                        

# model training and evaluation
# def train_predict_evaluate_model(classifier, 
#                                  train_features, train_labels, 
#                                  test_features, test_labels):
#     "simple evaluation matrix"
#     # build model    
#     classifier.fit(train_features, train_labels)
#     # predict using model
#     predictions = classifier.predict(test_features) 
#     # evaluate model prediction performance   
#     get_metrics(true_labels=test_labels, predicted_labels=predictions)
#     return predictions


def vectorize_terms(terms):
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term]) 
                for term in terms]
    return terms
    
def boc_term_vectors(word_list):
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(
                        np.hstack([list(word) 
                        for word in word_list]))
    word_list_term_counts = [{char: count for char, count in itemfreq(list(word))}
                             for word in word_list]
    
    boc_vectors = [np.array([int(word_term_counts.get(char, 0)) 
                            for char in unique_chars])
                   for word_term_counts in word_list_term_counts]
    return list(unique_chars), boc_vectors

def hamming_distance(u, v, norm=False):
    "calculate hamming distance"
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return (u != v).sum() if not norm else (u != v).mean()
    
def manhattan_distance(u, v, norm=False):
    "calculate manhattan distance"
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return abs(u - v).sum() if not norm else abs(u - v).mean()

def euclidean_distance(u,v):
    "calculate euclidean distance"
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    distance = np.sqrt(np.sum(np.square(u - v)))
    return distance


def levenshtein_edit_distance(u, v):
    "calculate levenshtein edit distance"
    
    # convert to lower case
    u = u.lower()
    v = v.lower()
    # base cases
    if u == v: return 0
    elif len(u) == 0: return len(v)
    elif len(v) == 0: return len(u)
    # initialize edit distance matrix
    edit_matrix = []
    # initialize two distance matrices 
    du = [0] * (len(v) + 1)
    dv = [0] * (len(v) + 1)
    
    # du: the previous row of edit distances
    for i in range(len(du)):
        du[i] = i
        
    # dv : the current row of edit distances    
    for i in range(len(u)):
        dv[0] = i + 1
        # compute cost as per algorithm
        for j in range(len(v)):
            cost = 0 if u[i] == v[j] else 1
            dv[j + 1] = min(dv[j] + 1, du[j + 1] + 1, du[j] + cost)
        # assign dv to du for next iteration
        for j in range(len(du)):
            du[j] = dv[j]
        # copy dv to the edit matrix
        edit_matrix.append(copy.copy(dv))
    
    # compute the final edit distance and edit matrix
    distance = dv[len(v)]
    edit_matrix = np.array(edit_matrix)
    edit_matrix = edit_matrix.T
    edit_matrix = edit_matrix[1:,]
    edit_matrix = pd.DataFrame(data=edit_matrix,
                               index=list(v),
                               columns=list(u))
    return distance, edit_matrix
    
def cosine_distance(u, v):
    "calculate cosine distance"
    distance = 1.0 - (np.dot(u, v) / 
                        (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v))))
                     )
    return distance


def swn_polarity(text):
    "Return a sentiment polarity: 0 = negative, 1 = positive"
 
    def clean_text(text):
        text = text.replace("<br />", " ")
        #text = text.decode("utf-8")
        return text

    sentiment = 0.0
    tokens_count = 0
    text = clean_text(text)
    lemmatizer = WordNetLemmatizer()
    
    # sentence tokenize
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        # word tokenizer
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = pp.penn_to_wn_tags(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            
            # word lemma
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            # synonyms
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            # polarity
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0


def tokenlist_to_strlist(tokenlist, splitter=' '):
    "turn a token list to str list"
    
    return splitter.join(tokenlist)




@fpn.FunctionNode
def get_pos_words(dictionary, word_type):
    """
        get lists of words of the given POS type from dictionary
        :param: d: list of dicionaries of POS
        :param: word_type: POS type or group of words (adj, noun, verb, adverb)
        :return: a list of all words of the word_type
    """
    
    def flatten_list_of_lists(word_list):
        # remove None
        word_list = [sublist for sublist in word_list if sublist is not None]
        # flatten the list
        word_list = [item.lower() for sublist in word_list for item in sublist]
        
        return word_list

    adj = ['JJ', 'JJR', 'JJS']
    noun = ['NN', 'NNS', 'NNP', 'NNPS']
    verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adverb = ['RB', 'RBR', 'RBS']
    
    if word_type == 'adj':
        poslist = adj
    elif word_type == 'noun':
        poslist = noun
    elif word_type == 'verb':
        poslist = verb
    elif word_type == 'adverb':
        poslist = adverb
    else:
        poslist = [word_type]
        
    word_list = [d.get(pos) for pos in poslist for d in dictionary]
    # remove None
    word_list = flatten_list_of_lists(word_list)

    return word_list

@fpn.FunctionNode
def map_dictlist(dict_list, f = lambda x: list(x.values())):
    """
        map the dictionary with the predicate
        :param: dict_list : list of dictionary
        :param: f: map function applied to each item in list
        :return: resultant list
    """
    result = map(f, dict_list)
    return list(result)

@fpn.FunctionNode    
def flatten_multiple_list_of_list(word_list):
    "flatten multiple list of list"
    return [flatten_list(sublist) for sublist in word_list]

@fpn.FunctionNode 
def flatten_list(wordlist):
    "flatten a single list of lists"
    return [item for sublist in wordlist for item in sublist]

@fpn.FunctionNode 
def flatten_strlist(strlist):
    "flatten a single list of lists"
    return [item for sentences in strlist]


# def extract_dict(df, col):
#     """
#         extract string json format data to json data 
#         :param: df: data frame
#         :param: col: name of column
#         :return: list of dictionary extracted from df
#     """
#     dictionary = df[col]
#     dictionary = [json.loads(item) for item in list(dictionary)] 
#     return dictionary

@fpn.FunctionNode 
def str2json(data):
    """
        extract string json format data to json data 
        :param: data: string formatted json data
        :return: list of dictionary extracted from df
    """

    data = [json.loads(item) for item in list(data)] 
    return data

@fpn.FunctionNode 
def filter_dict(dictionary, keys):
    """
        filter dictionary entries
        :param: dictionary: dict strucutre
        :param: keys: keys to retain
        :return: filtered dictionary 
    """
    new_dict = { k: dictionary.get(k) for k in dictionary.keys() if k in keys}
    return new_dict

@fpn.FunctionNode 
def map_filter_dict(dict_list, f):
    """
        filter list of dictionary entries
        :param: dict_list: list of dictionary
        :param: f: filter function
        :return: filtered dictionary     
    """
    
    return list(map(f, dict_list))

@fpn.FunctionNode 
def filter_df_dict_col(df, col, keys):
    """
        filter col in a data frame
        :param: df: data frame object
        :param: col: column name containing the dictionary
        :param: keys: list of keys to retain
    """
    
    sdf = extract_dict(df, col)
    f = partial(filter_dict, keys = keys)
    return list(map(f, sdf))


def filter_tokens(tokens, f):
    """
        extract all sentences containing the key word
        :param: list of tokens
        :param: targeted key word
        :return list of tokens contating key words
    """
    
    pass

def contains(token_list, w):
    """
        return true if w is one of the token in token list
        :param: token_list : list of tokens
        :param: w : key word
        :return: boolean
    """
    return w in token_list

@fpn.FunctionNode 
def get_unigrams(sent):
    """"
        return a list of bigrams from a sentence
        :params: sent: a string
        :return: list of tuples of bigram
    """
    result = [b for b in sent.split(" ")]
    return result


@fpn.FunctionNode 
def get_bigrams(sent):
    """"
        return a list of bigrams from a sentence
        :params: sent: a string
        :return: list of tuples of bigram
    """
    bigrams = [b for b in zip(sent.split(" ")[:-1], sent.split(" ")[1:])]
    return bigrams

@fpn.FunctionNode 
def get_trigrams(sent):
    """"
        return a list of bigrams from a sentence
        :params: sent: a string
        :return: list of tuples of bigram
    """
    trigrams = [b for b in zip(sent.split(" ")[:-2], sent.split(" ")[1:-1], sent.split(" ")[2:])]
    return trigrams


def map_tokens(f, token_list):
    "map a function to a list of tokens"
    return map(f, token_list)

def search_word(sentences, word, tokenizer=nltk.word_tokenize):
    """
        return index from the list of sentences containing word
        :param: sentences: list of string
        :param: word: target word
        :return: list of indices containing word
    """
    
    return [idx for idx, txt in zip((range(len(sentences))), sentences) if word in tokenizer(txt)]

def search_words(sentences, words, tokenizer=nltk.word_tokenize):
    """
        return index from the list of sentences containing word
        :param: sentences: list of string
        :param: words: list of words
        :param: tokenizer
        :return: list of indices containing word
    """
    
    return [idx for idx, txt in zip((range(len(sentences))), sentences) if set(words).issubset(tokenizer(txt))]


def get_sentences_with_matched_words(sentences, words):
    """
        return sentences with the bigram
    """
    
    # find sentences containing all the words
    idx = search_words(sentences, words)    
    sentences = [sentences[k] for k in idx]
    
    return sentences
    

def get_most_common_assoc_words(sentences, words, cutoff=0.05, num_return=20, norm=True, 
                                transformer=get_bigrams, tokenizer=nltk.word_tokenize):
    """
        return count (or weight) of bigrams containing the target tokens (words)
        :param: sentences : list of string
        :param: words: list of tokens
        :param: norm: boolean if normalize the count
        :param: tokenizer
        :return: most common bigrams in the sentences containing words
                    output format: [('word1', weight), ('word2', weight), ...]
    """

    # find sentences containing all the words
    idx = search_words(sentences, words)    
    sentences = [sentences[k] for k in idx]
    # convert all matched sentences to bigram list
    tmp = list(map(transformer, sentences))
    # flatten the list
    transformed_list = flatten_list(tmp)
    # set up a counter
    c = collections.Counter(transformed_list)
    
    # convert tuples into one word (e.g. look_like)
    result = [(join_tuples(tp), num) for tp, num in c.most_common(num_return)]
    # convert the count into proportion
    if norm == True:
        result = normalize_weight(result)
        
    return result

def normalize_weight(result):
    """
        normalize list of tuples
        :param: result: format [(word, weight), ...]
        :return: same format but weight normalized
    """
    
    total_weight = 0
    for item in result:
        total_weight += item[1]
    # create new list
    new_result = []
    for item in result:
        new_item = (item[0], item[1] / total_weight)
        new_result.append(new_item)
    
    return new_result
    

@fpn.FunctionNode     
def join_tuples(tp):
    "join string tuple to a single string"
    
    size = len(tp)
    result = ""
    if size > 0:
        result = tp[0]
        for k in range(size-1):
            result = result + "_" + tp[k+1]
        
    return result    
    
@fpn.FunctionNode     
def join_tuples_list(tp_list):
    "join string tuple to a single string"
    
    return list(map(join_tuples, tp_list))

    
# for bigrams
def get_word_dict(bigrams, content, cutoff=0.05, num_return=20, transformer=get_bigrams, norm=True):
    """
        create a dictionary of links
        :param: bigrams: starting bigrams [('bigram1', weight), ('bigram2', weight), ...]
        :param: content: list of sentences
        :param: cutoff: min weight of link to be included
        :param: num_return: num top topics
        :return: dicitonary in the following format: 
                    {w0: [(w1, weight), ... ], w1: [...]}
        
    """
    # all words found so far
    founded_words = set()
    dictionary = {}

    for item in bigrams:
        # check weight
        weight = item[1]
        word = item[0]   # a bigram
        if (word not in founded_words) and (weight > cutoff):
            # split the bigrams into word tokens
            tokens = word.split("_")
            # get list of top associated words with weight 
            bigram_list = get_most_common_assoc_words(content, tokens, transformer=transformer, num_return=num_return, norm=norm)
            # 
            (founded_words, dictionary) = add_link_dict(word, bigram_list, cutoff, founded_words, dictionary)

    return dictionary



def add_link_dict(word, bigram_list, cutoff, founded_words, dictionary):
    """
        add link dictionary of links
        :param: word: bigram word
        :param: bigrams: list of bigrams and weights [('xxx', v), ('yyy', n), ...]
        :param: cutoff: min weight of link to be included
        :return: dicitonary in the following format: 
                    {w0: [(w1, weight), ... ], w1: [...]}
    """
    
    def link_exists(word, link_word, dictionary):
        "find if link exists"

        if word == link_word:
            return True
        
        # find entry of word
        keys = dictionary.keys()
        if word in keys:
            linked_list = dictionary.get(word)
            for item in linked_list:
                if item[0] == link_word:
                    return True
        
        if link_word in keys:
            linked_list = dictionary.get(link_word)
            for item in linked_list:
                if item[0] == word:
                    return True        
            
        return False

    # start to craw word
    founded_words.add(word)
    for item in bigram_list:
        weight = item[1]
        if weight > cutoff:
            link_word = item[0]
            # if link is not recorded
            if not link_exists(word, link_word, dictionary):
                # need only to add one entry
                if word in dictionary.keys():
                    dictionary[word].append(item)
                else:
                    dictionary[word] = [item]
                
                # check if needed to crawl next
                if link_word not in founded_words:
                    (founded_words, dictionary) = add_link_dict(link_word, bigram_list, cutoff, founded_words, dictionary)

    return (founded_words, dictionary)


def parse_network_dict(d):
    "convert the network graph to json"
    
    def key_exists(node, nodelist):
        "check if node exists"
        
        for item in nodelist:
            if node == item["id"]:
                return True
        
        return False
    
    data = {}
    data['nodes'] = []
    data['links'] = []
    
    # parse linked graph
    for key in d.keys():
        linked_list = d[key]
        if not key_exists(key, data["nodes"]):
            # FIXME: fixed node value (circle size)
            # FIXME: group??
            item = {"id": key, "group": 1, "value": 5}
            data["nodes"].append(item)    
        for tp in linked_list:
            # lined value
            item = {"source": key, "target": tp[0], "value": tp[1]}
#             item = {"source": key, "target": tp[0], "value": 5}
            data['links'].append(item)
            # add node item
            if not key_exists(tp[0], data["nodes"]):
                # FIXME: fixed node value (circle size)
                item = {"id": tp[0], "group": 1, "value": 5}
                data["nodes"].append(item)
                
    return data

def values_exists_in_list(source_list, target_list):
    """
        find if the values is in one of item in target list
        :param: source_list : list of values
        :param: target_list : list of targeted values
        :return: list of boolean
    """
    
    return list(map(lambda x: x in target_list, source_list))

# def create_bag_of_centroids( wordlist, word_centroid_map ):
#     "bag of centroids"
#     
#     # The number of clusters is equal to the highest cluster index
#     # in the word / centroid map
#     num_centroids = max( word_centroid_map.values() ) + 1
#     
#     # Pre-allocate the bag of centroids vector (for speed)
#     bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
#     
#     # Loop over the words in the review. If the word is in the vocabulary,
#     # find which cluster it belongs to, and increment that cluster count
#     # by one
#     for word in wordlist:
#         if word in word_centroid_map:
#             index = word_centroid_map[word]
#             bag_of_centroids[index] += 1
#     
#     # Return the "bag of centroids"
#     return bag_of_centroids
