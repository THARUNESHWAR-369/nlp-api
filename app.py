from fastapi import FastAPI, Query


import pickle
import os
from typing import List
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

app = FastAPI()



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
        __tokenize_words = self.__do_tokenize(self.__text)
        __stopwords = self.__do_stopword(__tokenize_words)
        __stemming_words = self.__do_stemming(__stopwords)
        __lemmentize_words = self.__do_lemmentize(__stemming_words)
        
        __preprocessed_text = " ".join([i for i in __lemmentize_words])
        print("from preprocessing: ",__preprocessed_text)
        return __preprocessed_text if len(__preprocessed_text) > 2 else None
    

class SENTIMENT_ANALYSIS(NLP_PREPROCESS):
    
    __DIR = "models/ml-models/"
    __MODEL_PATH = f"{__DIR}/LinearSVCV1.pkl"
    __VECTOR_PATH = f"{__DIR}/vectors/countVectorV1.pkl"
    
    def __init__(self, text: str = None, textList: list = None) -> None:
        super().__init__(text)
        self.__text = text
        self.__textList = textList
        
        print(os.getcwd())
        
        self.__loadModel()
        
    def __mapToString(self, prediction) -> str:
        if prediction == 0:
            return "NEGATIVE"
        elif prediction == 1:
            return "NEUTRAL"
        else:
            return "POSITIVE"
        
    def __loadModel(self) -> None:
        self.__MODEL = pickle.load(open(self.__MODEL_PATH, 'rb'))
        self.__VECTORIZER = pickle.load(open(self.__VECTOR_PATH, 'rb'))
        
    def getSentiment(self) -> str:
        __preprocessed_text = self.preprocess()
        return {
            "text": self.__text,
            #"preprocessed-text": __preprocessed_text  ,
            "sentiment":  self.__mapToString(prediction = self.__MODEL.predict(
                self.__VECTORIZER.transform([self.__text])
            ))
        }
        
    def getMultipleSentiment(self) -> dict:
        __ALL_SENTIMENT = {
            "sentiments":[]
        }
        for text in self.__textList:
            self.__text = text
            __ALL_SENTIMENT["sentiments"].append(self.getSentiment())
        
        return __ALL_SENTIMENT

@app.get("/")
def home():
    return 503

@app.get("/api/v1")
def api():
    return 503

@app.get("/api/v1/sentiment-analysis/")
async def sentiment(text: str):
    sentiment_obj = SENTIMENT_ANALYSIS(text=text)
    return sentiment_obj.getSentiment()
    
@app.get("/api/v1/sentiment-analysis/multi-text/")
async def sentiment(text: List[str] = Query(None)):
    sentiment_obj = SENTIMENT_ANALYSIS(textList=text)
    print(sentiment_obj.getMultipleSentiment())
    return sentiment_obj.getMultipleSentiment()
