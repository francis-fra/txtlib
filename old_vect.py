#---------------------------------------------------------
# FIXME: to be replaced
#---------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import TransformerMixin
import numpy as np    
import os

# from txtlib import transform as ttf
from gensim import corpora

# FIXME: gensim vectorizer??
class TextVectorizer(TransformerMixin):
    
    def __init__(self, corpus=None):
        # list of document (free text)
        self.corpus = corpus
        
    def init_bow_extractor(self, corpus=None, **kwargs):
        "returns bag of words vectorizer"
        
        if corpus is None:
            corpus = self.corpus
            
        self.vectorizer = CountVectorizer(**kwargs)
        self.features = self.vectorizer.fit_transform(corpus)
                
    def get_vectorizer(self):
        return self.vectorizer
    
    def get_features(self):
        return self.features

    def init_tfidf_extractor(self, corpus=None, **kwargs):
        "tf-idf vectorizer"
        
        if corpus is None:
            corpus = self.corpus
            
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.features = self.vectorizer.fit_transform(corpus)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        "do transform"
        return (self.vectorizer.fit_transform(X))
        
    def get_tfidf_features(self, matrix, **kwargs):
        "convert a count matrix to a normalized tf or tf-idf representation"
        transformer = TfidfTransformer(**kwargs)
        return (transformer.transform(matrix))

# FIXME: 
class CorpusGenerator(object):
    "memory efficient bag of words generator"
    def __init__(self, x):
        '''
            :param:    strlist: list of string
        '''
        
        self.x = x
        transformer = ttf.TextTransformer()
        xx = [transformer.tokenize(sent) for sent in x]
        self.dictionary = corpora.Dictionary(xx)
    
    def __iter__(self):
        "generate bag of words vectors"
        for line in self.x:
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())


class CorpusLoader(object):
    "crawl the whole directory for training data set"
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        "read line by line"
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line



# def average_word_vectors(words, model, vocabulary, num_features):
#     ""
#     feature_vector = np.zeros((num_features,),dtype="float64")
#     nwords = 0.
#     
#     for word in words:
#         if word in vocabulary: 
#             nwords = nwords + 1.
#             feature_vector = np.add(feature_vector, model[word])
#     
#     if nwords:
#         feature_vector = np.divide(feature_vector, nwords)
#         
#     return feature_vector
#     
#    
# def averaged_word_vectorizer(corpus, model, num_features):
#     ""
#     vocabulary = set(model.wv.index2word)
#     features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
#                     for tokenized_sentence in corpus]
#     return np.array(features)
#     
#     
# def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
#     
#     word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] 
#                    if tfidf_vocabulary.get(word) 
#                    else 0 for word in words]    
#     word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
#     
#     feature_vector = np.zeros((num_features,),dtype="float64")
#     vocabulary = set(model.wv.index2word)
#     wts = 0.
#     for word in words:
#         if word in vocabulary: 
#             word_vector = model[word]
#             weighted_word_vector = word_tfidf_map[word] * word_vector
#             wts = wts + word_tfidf_map[word]
#             feature_vector = np.add(feature_vector, weighted_word_vector)
#     if wts:
#         feature_vector = np.divide(feature_vector, wts)
#         
#     return feature_vector
#     
# def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors, 
#                                    tfidf_vocabulary, model, num_features):
#                                        
#     docs_tfidfs = [(doc, doc_tfidf) 
#                    for doc, doc_tfidf 
#                    in zip(corpus, tfidf_vectors)]
#     features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
#                                    model, num_features)
#                     for tokenized_sentence, tfidf in docs_tfidfs]
#     return np.array(features)


