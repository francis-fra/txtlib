from txtlib import utility as ut
from txtlib import preprocessor as pp
import collections
import json
import pprint
import gensim
import gensim.corpora as corpora
import nltk
import spacy
import pandas as pd
import numpy as np
from functools import partial
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import confusion_matrix

# for bigram
def generate_network_graph(content, save_file_name, cutoff=0.05, num_return=20, norm=True):
    """
        generate json file for d3 graph
    """
    
    make_bigram_list = ut.get_bigrams >> ut.join_tuples_list
    bigram_list = pp.normalize_corpus(content, make_bigram_list)
    
    # find the most frequent bigrams for the whole corpus
    c = collections.Counter(ut.flatten_list(bigram_list))
    
    common_bigrams = c.most_common(num_return)
    if norm == True:
        common_bigrams = ut.normalize_weight(common_bigrams)
       
    result = ut.get_word_dict(common_bigrams, content, cutoff=cutoff, num_return=num_return, norm=norm)
    data = ut.parse_network_dict(result)
    
    with open(save_file_name, 'w') as outfile:
        json.dump(data, outfile)
        
    outfile.close()        
    print("File saved to %s\n" % save_file_name)

def generate_keyword_graph(content, keywords, save_file_name, cutoff=0.05, num_return=20, norm=True):
    """
        generate json file for d3 graph
    """
    
    leads = []
    if norm == True:
        value = 1
    else:
        value = 1e10
    for word in keywords:
        leads.append((word, value))

    result = ut.get_word_dict(leads, content, cutoff=cutoff, num_return=num_return, norm=norm)
    data = ut.parse_network_dict(result)
    
    with open(save_file_name, 'w') as outfile:
        json.dump(data, outfile)
        
    outfile.close()        
    print("File saved to %s\n" % save_file_name)
    

def get_counter_example(matched_idx, data, y_true, label):
    """
        get counter example of the unmatched content
        
        :param: matched_idx: index which are matched
        :param: data: list of data to be returned
        :param: y_true: true label list
        :param: label: targeted label
        :return: a slice of the data where prediction is 
                    different from y_true where y = label
    """
    
    true_idx = pd.Series(y_true) == label
    idx = [a and (not b) for a, b in zip(true_idx, matched_idx)]
    return pd.Series(data)[idx]
    

def evaluate_sentiment(txt_list, y_true):
    """
        Return confusion matrix of sentiment prediction
        :param: txt_list : raw text
        :param: y_true: numerical sentiment value
        :return: confusion matrix 
    """
    
    y_pred = get_sentiment_list(txt_list)
    y_true = list(map(num_to_sign, y_true))
    print (confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]))
    idx = (np.array(y_true) == np.array(y_pred)).tolist()
    return idx
#     return [n for n, matched in zip(range(len(idx)), idx) if matched == False]
    

def num_to_sign(num):

    if num > 0:
        num = 1
    elif num < 0:
        num = -1
    else:
        num = 0

    return num

def get_sentiment(txt, sign=True):
    "get sentiment scores"
    
    polarity = TextBlob(txt).sentiment.polarity
    if sign == True:
        polarity = num_to_sign(polarity)
    
    return polarity 

def get_sentiment_list(txt_list, sign=True):
    "get sentiment scores"
    
    f = partial(get_sentiment, sign=sign)
    return list(map(f, txt_list))


def build_LDA_model(freeText):
    "to be removed"
    f = pp.fillNaN >> pp.lemmatize >> pp.yield_token
    out = pp.normalize_corpus(freeText, f)
    data_words = pp.get_words_from_corpus(out)
    
    # yield generator!!
    bigram_mod = pp.get_bigrams_mod(data_words)
    #trigram_mod = pp.get_trigrams_mod(data_words, bigram_mod)
    
    data_words_bigrams = pp.make_bigrams(data_words, bigram_mod)
    #data_words_trigrams = pp.make_trigrams(data_words, bigram_mod, trigram_mod)
    
    nlp = spacy.load('en', disable=['parser', 'ner'])
    clean_content = pp.lemmatization(data_words_bigrams, nlp)
    #clean_content = pp.lemmatization(data_words_trigrams, nlp)
    clean_content = list(clean_content)
    
    id2word = corpora.Dictionary(clean_content)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in clean_content]
    # corpus
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    return (lda_model, corpus, id2word)

class GensimModel(object):
    def __init__(self, corpus):
        self.raw_corpus = corpus
        self.corpus = None
        self.tokens = None
        self.id2word = None
        self.idamodel = None
        self.tfidfmodel = None
        self.bow = None
        self.vis = None

    def tokenize_corpus(self, txtlist=None):
        "tokenize list of string"
        if txtlist is None:
            txtlist = self.corpus
        self.tokens = [self.tokenize(sent) for sent in txtlist]
        return self

    def tokenize(self, txt, f=nltk.word_tokenize):
        "tokenize text"
        tokens = f(txt) 
        tokens = [token.strip() for token in tokens]
        return tokens
    
    def clean(self, f):
        "clean corpus with the supplied cleaning function"
        self.corpus = pp.normalize_corpus(self.raw_corpus, f)
        return self
    
    def create_dictionary(self, tokens=None):
        "create word dictionary"
        if tokens is None:
            tokens = self.tokens
        
        self.id2word = corpora.Dictionary(tokens)
        return self
        
    def get_bow(self, tokens=None):
        "bag of words"
        if tokens is None:
            tokens = self.tokens
        self.bow = [self.id2word.doc2bow(txt) for txt in tokens]
        return self
        
    def print_topics(self):
        "show topics"
        pprint(self.model.print_topics())
        return self
    
    # TODO: tfidf
    def trainTFIDF(self):
        if self.bow is not None:
            self.tfidfmodel = gensim.models.TfidfModel(corpus=self.bow)
        return self
    
    def apply(self, token):
        if self.tfidfmodel is not None:
            return self.tfidfmodel[token]
        
    def trainIDA(self, num_topics=10):
        "train topic model"
        # need to vectorize and craete word dictionary first
        if self.bow is not None and self.id2word is not None:
            self.idamodel = gensim.models.ldamodel.LdaModel(corpus=self.bow,
                                                           id2word=self.id2word,
                                                           num_topics=num_topics)
        return self
    
    def prepare(self):
        "prepare data for visualization"
        self.vis = pyLDAvis.gensim.prepare(self.idamodel, self.bow, self.id2word)
        return self
        
    def show(self):
        "show in broswer"
        if self.vis is None:
            self.prepare()
        pyLDAvis.show(self.vis)
        return self

    def save(self, outfile):
        "save model to html"
        if self.vis is None:
            self.prepare()
        pyLDAvis.save_html(self.vis, outfile)
        return self






