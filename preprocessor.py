import nltk, re, string
import function_pipe as fpn
import itertools
from sklearn.base import TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import gensim, spacy
from html.parser import HTMLParser
wnl = WordNetLemmatizer()
# from contractions import CONTRACTION_MAP
# FIXME:
from .contractions import CONTRACTION_MAP

@fpn.FunctionNode 
def fillNaN(txt, value=""):
    "fill with value if empty"
    return (value if str(txt) == 'nan' else txt)
        
@fpn.FunctionNode        
def yield_tokenize_list(txtlist):
    '''
        Tokenize list of text
        :param txt: list of text
        :return : list of tokens
    '''
        
    for sent in txtlist:
        yield tokenize(sent)

@fpn.FunctionNode
def tokenize_list(txtlist):
    '''
        Tokenize list of text
        :param txt: list of text
        :return : list of tokens
    '''
        
    return [tokenize(sent) for sent in txtlist]

@fpn.FunctionNode
def tokenize(txt, f=nltk.word_tokenize):
    '''
        Annotate text tokens with POS tags
        :param txt: free text
        :param f: tokenize function
        :return: list of tokens
    '''
                       
    tokens = f(txt) 
    tokens = [token.strip() for token in tokens]
    return tokens

@fpn.FunctionNode
def yield_token(txt, f=nltk.word_tokenize):
    '''
        Annotate text tokens with POS tags
        :param txt: free text
        :param f: tokenize function
        :return: list of tokens
    '''
                       
    tokens = f(txt) 
    tokens = [token.strip() for token in tokens]
    yield tokens

@fpn.FunctionNode
def lemmatize(input, f=wnl.lemmatize, toList=False):
    '''
        lemmatize and tag text
        :param :   input: free text or list of tokens
        :param :   f    : lemmatize function
        :return:   free text
        
        other popular stemmer: nltk.stem.porter.PorterStemmer
    '''
    # tokenize if input is a string
    if isinstance(input, str):
        tokens = tokenize(input)
    else:
        tokens = input
    
    
    pos_tagged_text = tag_pos(tokens)
    lemmatized_tokens = [f(word, pos_tag) 
                         if pos_tag else word for word, pos_tag in pos_tagged_text]
    
    if toList == False:
        return (' '.join(lemmatized_tokens))
    else:
        return lemmatized_tokens
    
@fpn.FunctionNode
def split_sentences(txt, f=nltk.tokenize.sent_tokenize):
    """
        split free txt into list of sentences
        :param :     input: free text
        :return:     token of sentences
    """
    return (f(txt))

@fpn.FunctionNode
def tag_pos(input, f=nltk.pos_tag, wn_tags=True):
    '''
        Annotate text tokens with POS tags
        :param tokens: list
        :param f: tagging function
        :param wn_tags: word net tag
        :return: string list
    '''
    
    def penn_to_wn_tags(pos_tag):
        '''
            convert from penn tag to wordnet tag
            :param     pos_tag: tag identifier (string)
            :return:   word net tag (string)  
        '''
        
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
     
    if isinstance(input, str):
        tokens = tokenize(input)
    else:
        tokens = input
                 
    tagged_text = f(tokens)
    
    # list of tuples (word, wn tag)
    if wn_tags:
        tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                             for word, pos_tag in tagged_text]
    else:
        tagged_lower_text = [(word.lower(), pos_tag) for word, pos_tag in tagged_text]
    
    return tagged_lower_text

@fpn.FunctionNode
def remove_pattern(txt, pattern="[^a-zA-Z]"):
    """
        remove characters from free text using regular expression
        :param     input: free text
        :param     filtered free text
    """
    txt = re.sub(pattern," ", txt)
    return txt

@fpn.FunctionNode
def remove_special_characters(txt):
    """
        remove characters from free text using regular expression
        :param     input: free text
        :param     filtered free text
    """
    punctuation = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return remove_pattern(txt, punctuation)


@fpn.FunctionNode
def beautify(txt):
    """
        remove non-alnum
        :param     txt: free text
        :return    filtered free text
    """
    tokens = tokenize(txt)
    tokens = [tok for tok in tokens if tok[0].isalnum()]
    return ' '.join(tokens)
    
@fpn.FunctionNode
def remove_stopwords(input, 
                    stopwords=nltk.corpus.stopwords.words('english'),
                    toList=False):
    '''
        Remove stop words
        :param input: either free text or token list
        :param stopwords: list of stop words
        :return: either free text or list of tokens
    '''
            
    if isinstance(input, str):
        tokens = tokenize(input)
    else:
        tokens = input
        
    filtered_tokens = [token for token in tokens if token not in stopwords]
    
    if toList == False:
        return (' '.join(filtered_tokens))
    else:
        return filtered_tokens
    
@fpn.FunctionNode
def strip_html_tags(txt):
    "strip html tags"
    
    class MLStripper(HTMLParser):
        def __init__(self):
            self.reset()
            self.strict = False
            self.convert_charrefs= True
            self.fed = []
        def handle_data(self, d):
            self.fed.append(d)
        def get_data(self):
            return ''.join(self.fed)

    s = MLStripper()
    s.feed(txt)
    
    return s.get_data()
    
@fpn.FunctionNode
def expand_contractions(txt, contraction_mapping=CONTRACTION_MAP):
    "replace contraction to full word in the text"
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, txt)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

@fpn.FunctionNode        
def lowerCase(txt):
    "pipeling the lower case conversion"
    return txt.lower()
    
def normalize_txt(txt, strip_html=False, islemmatize=True, istokenize=False):
    """
        normalize raw text
        :param: txt : free text
        :param: free txt or token           
    """

    f = beautify
    
    if strip_html:
        f = f >> strip_html_tags
                              
    if islemmatize:
        f = f >> lemmatize       
    else:
        f = f >> lowerCase
    
    f = f >> remove_stopwords
    
    if istokenize:
        f = f >> tokenize
            
    return f(txt)

#----------------------------------------------------------------
# corpus methods
#----------------------------------------------------------------
# the following works with list of list of tokens 

def normalize_corpus(corpus, f=normalize_txt, **kwargs):
    '''
        normalize corpus by calling normalize txt
        :param: corpus : list of free text
        :param: others param are same as normalize_txt
    '''
    
    return [f(txt, **kwargs) for txt in corpus]
            
def get_words_from_corpus(corpus):
    '''
        get all words from corpus
        :param: corpus : list of list of tokens
        :return: list of tokens 
    '''
    return list(itertools.chain.from_iterable(corpus))

# create bigrams and trigrams models
def get_bigrams_mod(data_words, min_count=5, threshold=100):
    "get bigram function"
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=threshold)
    return gensim.models.phrases.Phraser(bigram)
    
def get_trigrams_mod(data_words, bigram, threshold=100):
    "get trigram function"
    trigram = gensim.models.Phrases(bigram[data_words], threshold=threshold)
    return gensim.models.phrases.Phraser(trigram)

def make_bigrams(corpus, bigram_mod):
    """
        generate bigrams
        :param: corpus : list of list of tokens
        :return: yield list of bigram tokens
    """
    for doc in corpus:
        yield bigram_mod[doc]
#     return [bigram_mod[doc] for doc in corpus]

def make_trigrams(corpus, bigram_mod, trigram_mod):
    """
        generate trigrams
        :param: corpus : list of list of tokens
        :return: yield list of trigram tokens
    """
    for doc in corpus:
        yield trigram_mod[bigram_mod[doc]]
#     return [trigram_mod[bigram_mod[doc]] for doc in corpus]

def lemmatization(corpus, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
        spacy lemmatize
        :param: corpus : list of list of tokens
        :param: nlp : spacy nlp
        :return yield list of list of tokens 
    """
 
    #nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in corpus:
        doc = nlp(" ".join(sent)) 
        yield [token.lemma_ for token in doc if token.pos_ in allowed_postags]





        