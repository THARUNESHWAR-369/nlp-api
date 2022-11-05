
try:
    import nltk
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
except:
    import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import string

class NLP_PREPROCESS:
    
    __PORTER_STEMMER = PorterStemmer()
    __LEMMATIZER = WordNetLemmatizer()
    
    __PUNCTUATIONS = string.punctuation
    
    def __init__(self, text) -> None:
        self.__text = text
        
        self.__STOPWORDS = set(stopwords.words('english'))
        for p in string.punctuation:
            self.__STOPWORDS.add(p)
            
    def __do_tokenize(self, sentence) -> list:
        return word_tokenize(sentence)

    def __do_stopword(self, tokenized_words) -> list:
        return [w for w in tokenized_words if not w in self.__STOPWORDS]
    
    def __do_stemming(self, filterd_words) -> list:
        return [self.__PORTER_STEMMER.stem(fw) for fw in filterd_words]
    
    def __do_lemmentize(self, stem_words) -> list:
        return [self.__LEMMATIZER.lemmatize(sw) for sw in stem_words]
    
    def preprocess(self):
        print("self.__text: ",self.__text)
        __tokenize_words = self.__do_tokenize(self.__text)
        print("__tokenize_words: ",__tokenize_words)
        __stopwords = self.__do_stopword(__tokenize_words)
        __stemming_words = self.__do_stemming(__stopwords)
        __lemmentize_words = self.__do_lemmentize(__stemming_words)
        
        __preprocessed_text = " ".join([i for i in __lemmentize_words])
        return __preprocessed_text if len(__preprocessed_text) > 2 else None