import spacy
import nltk
from collections import defaultdict 
from collections import Counter
from numpy.linalg import norm
from numpy import dot
import gensim
from gensim.models import word2vec
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS


# modelDir = '/home/fra/DataMart/datacentre/corpus/'
modelDir = 'H:\\Documents\\wiki\\'
# googleVectors = modelDir + 'GoogleNews-vectors-negative300.bin' 
commonCrawlVectors = modelDir + 'CommonCrawl.42B.300d.txt' 

# corpusDir = '/home/fra/DataMart/datacentre/corpus/data/'
corpusDir = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\text_mining\\corpus\\'
text8corpus = corpusDir + 'text8'


# def word_count(corpus):
#     pass

class GensimProcessor(object):
    "Gensim Word2vec processor"
    
    def __init__(self):
        pass
    
    def load_Text8Corpus(self, corpus=text8corpus):
        "load corpus from disk"
        sentences = word2vec.Text8Corpus(text8corpus)
        self.model = word2vec.Word2Vec(sentences, size=200, hs=1, negative=0)
    
    def load_word2vec_model(self, model=commonCrawlVectors):
        "load pretrained vectors"
        if model[-3:] == 'txt':
            isBinary = False
        else:
            isBinary = True
        print("INFO: loading...")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=isBinary)
        print('INFO: Done!')

    def get_analogy(self, **kwargs):
        "get analogy from positive and negative"
        
        return (self.model.wv.most_similar(**kwargs))
        
    def get_odd_one_out(self, txt):
        "get odd one out"
        
        if isinstance(txt, list):
            tokens = txt
        else:
            tokens = txt.split()
        
        return (self.model.wv.doesnt_match(tokens))
    
    def get_vector(self, word):
        "get word vector"
        
        return (self.model.wv[word])
    
    def get_score(self, txt):
        "probability of a text under the model"
        
        if isinstance(txt, list):
            tokens = txt
        else:
            tokens = txt.split()
                    
        return (self.model.score(tokens))
    
    def get_word_frequency(self, document, stopwords=nltk.corpus.stopwords.words('english'), rmstopwords=True, num=10):
        "get word counter in the document"
        
        if rmstopwords:
            tokens = [token.lower() for token in gensim.utils.tokenize(document) if token.lower() not in stopwords]
        else:
            tokens = [token.lower() for token in gensim.utils.tokenize(document)]
            
        return (Counter(tokens).most_common(num))

    
    def tokenize(self, document, **kwargs):
        "tokenize document"
        
        return (list(gensim.utils.tokenize(document, **kwargs)))
        

    # TODO:
    # word sim
#     model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
# bigram_transformer = gensim.models.Phrases(sentences)
# model = Word2Vec(bigram_transformer[sentences], size=100, ...)


class SpacyProcessor(object):
    "spacy text processor"
    def __init__(self, model = 'en'):
        self.nlp = spacy.load(model)
        
    def get_tokens(self, txt):
        '''
            :return spacy doc object
        '''
        return self.nlp.tokenizer(txt)
        
    def get_similarity(self, txt01, txt02):
        "get similarity score of two text"
        
        if isinstance(txt01, str):
            txt01 = self.nlp.tokenizer(txt01)
            
        if isinstance(txt02, str):
            txt02 = self.nlp.tokenizer(txt02)
            
        return txt01.similarity(txt02)
    
    def get_most_similar(self, word, num=15):
        "list the most similar words"
        
        if isinstance(word, str):
            word = self.nlp.vocab[word]
            
        # gather all known words, take only the lowercased versions
        cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
        
        allWords = list({w for w in self.nlp.vocab if w.has_vector and w.orth_.islower() and w.lower_ != word.text.lower()})
        allWords.sort(key=lambda w: cosine(w.vector, word.vector))
        allWords.reverse()
        topList = []
        for word in allWords[:num]:   
            topList.append(word.orth_)
            
        return (topList)
    
    def print_txt_info(self, txt):
        "print basic info of the text"
        
        doc = self.nlp(txt)
        for word in doc:
            print("-" * 40)
            print("Text   : {}".format(word.text))
            print("Lemma  : {}".format(word.lemma_))
            print("Tag    : {}".format(word.tag_))
            print("POS    : {}".format(word.pos_))
            
    def split_sentences(self, document):
        "split document into sentences"
        
        document = self.nlp(document)
        return (list(document.sents))
    
    def get_analogy(self, add=None, subtract=None, num=10):
        "show the result of the analogy"
        
        if add is not None:
            positive_vec = self.nlp.vocab[add[0]].vector
            
        for word in add[1:]:
            positive_vec += self.nlp.vocab[word].vector
            
        if subtract is not None:
            negative_vec = self.nlp.vocab[subtract[0]].vector
            
        for word in subtract[1:]:
            negative_vec += self.nlp.vocab[word].vector
            
        result = positive_vec - negative_vec
        
        # gather all known words, take only the lower cased versions
        allWords = list({w for w in self.nlp.vocab 
                         if w.has_vector and w.orth_.islower() and w.lower_ not in add and w.lower_ not in subtract})

        # sort by similarity to the result
        cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
        allWords.sort(key=lambda w: cosine(w.vector, result))
        allWords.reverse()
        
        return ([word.orth_ for word in allWords[:num]])
    
    def _cleanup(self, token, lower=True):
        "strip token"
        if lower:
            token = token.lower()
        return token.strip()
            
    def get_entity_dict(self, document):
        '''
            :return {entity: [list of labels]}
        '''

        # all types of entities
        document = self.nlp(document)
        labels = set([w.label_ for w in document.ents])
    
        d = defaultdict(list)
        # assign entities to dict 
        for label in labels: 
            entities = [self._cleanup(e.string) for e in document.ents if label==e.label_] 
            entities = list(set(entities))
            d[label].append(entities)
            
        return d
    
    def get_word_freq(self, document, f=lambda x: True, num=10):
        '''
            return the most freq words
            :param document : list of tokens
            :param f        : token filter
            :param num      : num words to output
            :return         : counter of words
        '''
        
        document = self.nlp(document)
        cleaned_list = [self._cleanup(word.string) for word in document if f(word)]
        return (Counter(cleaned_list).most_common(num))
                
    # FIXME: same as preprocessor method
    def lemmatize(self, corpus, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """
            lemmatize corpus
            :param: corpus : list of list of tokens
            :param: nlp : spacy nlp
            :return yield list of list of tokens 
        """

        for sent in corpus:
            doc = self.nlp(" ".join(sent)) 
            yield [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        
        
        