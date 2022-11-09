import pickle

from .text_preprocessor import NLP_PREPROCESS

from modules.load_model import LOAD_MODEL

import numpy as np

class BaseModel:
    
    def __init__(self, text, languageData, sentimentData, textEmotionData) -> None:
        self.__TEXT = text
        self.__LANGUAGE = languageData
        self.__SENTIMENT = sentimentData
        self.__TEXT_EMOTION = textEmotionData
        #self.__CONFIDENCE = confidence
        
    def to_json(self):
        return {
            "text" : self.__TEXT,
            "text-sentiment" : self.__SENTIMENT,
            "language" : self.__LANGUAGE,
            "text-emotion" : self.__TEXT_EMOTION
        }

class SENTIMENT_ANALYSIS(NLP_PREPROCESS):
    
    def __init__(self, text: str = None, textList: list = None, languageDetectionModelVersion: str = 'v2', sentimentAnalysisModelVersion: str = 'v3', textEmotionDetectionModelVersion: str = 'v1') -> None:
        super().__init__(text)
        self.__text = text
        self.__textList = textList
        self.__language_detection_model_version = languageDetectionModelVersion
        self.__sentiment_analysis_model_version = sentimentAnalysisModelVersion
        self.__text_emotion_detection_model_version = textEmotionDetectionModelVersion
                
        self.__loadModel()
        
    def __mapToString(self, prediction) -> str:
        if prediction == 0:
            return "NEGATIVE"
        elif prediction == 1:
            return "NEUTRAL"
        else:
            return "POSITIVE"
    
        
    def __loadModel(self) -> None:
        __TEXT_EMOTION_MODELS = LOAD_MODEL(base_type='text-emotion-detection-model', version=self.__text_emotion_detection_model_version).load()
        
        __LANGUAGE_MODELS = LOAD_MODEL(base_type="language-detection-model", version=self.__language_detection_model_version).load()
        
        __SENTIMENT_MODELS = LOAD_MODEL(base_type="sentiment-analysis-model", version=self.__sentiment_analysis_model_version).load()
    
        
       # print(__TEXT_EMOTION_MODELS)
        
        self.__SEMTIMENT_VECTORIZER = __SENTIMENT_MODELS['vectorizer']
        self.__SENTIMENT_MODEL = __SENTIMENT_MODELS['model']
        
        self.__LANGUAGE_LABEL_ENCODER = __LANGUAGE_MODELS['labelEncoder']
        self.__LANGUAGE_MODEL  = __LANGUAGE_MODELS['model']
        self.__LANGUAGE_VECTORIZER = __LANGUAGE_MODELS['vectorizer']
        
        self.__TEXT_EMOTION_LABEL_ENCODER = __TEXT_EMOTION_MODELS['labelEncoder']
        self.__TEXT_EMOTION_MODEL  = __TEXT_EMOTION_MODELS['model']
        self.__TEXT_EMOTION_VECTORIZER = __TEXT_EMOTION_MODELS['vectorizer']
        
        #print(self.__TEXT_EMOTION_LABEL_ENCODER.classes_)
        
        
    def __getTextEmotion(self) -> dict:
        
        __TRANSFORM_TEXT = self.__TEXT_EMOTION_VECTORIZER.transform([self.__text])
        __PRED = self.__TEXT_EMOTION_MODEL.predict(__TRANSFORM_TEXT)
        __EMOTION = self.__TEXT_EMOTION_LABEL_ENCODER.inverse_transform(__PRED)[0]
        
        __CONFIDENCE = "{:.6f}".format(float(np.amax(self.__TEXT_EMOTION_MODEL.predict_proba(__TRANSFORM_TEXT)[0])))
        
        return {
            "emotion":__EMOTION.upper(),
            "confidence":__CONFIDENCE
        }
        
    def __getLanguage(self) -> dict:
        
        __TRANSFORM_TEXT = self.__LANGUAGE_VECTORIZER.transform([self.__text]).toarray()
        __PRED = self.__LANGUAGE_MODEL.predict(__TRANSFORM_TEXT)
        __LANGUAGE = str(self.__LANGUAGE_LABEL_ENCODER.inverse_transform(__PRED)[0]).upper()
        __CONFIDENCE = "{:.6f}".format(float(np.amax(self.__LANGUAGE_MODEL.predict_proba(__TRANSFORM_TEXT)[0])))
            
        return {
            "lang" : __LANGUAGE,
            "confidence" : __CONFIDENCE
        }
        
        
    def __getSentiment(self) -> dict:
        
        __TRANSFORM_TEXT = self.__SEMTIMENT_VECTORIZER.transform([self.__text])
        __PRED = self.__SENTIMENT_MODEL.predict(__TRANSFORM_TEXT)
        __SENTIMENT = self.__mapToString(
                prediction = __PRED
        )
        
        __SENTIMENT_CONFIDENCE = "{:.6f}".format(float(np.amax(self.__SENTIMENT_MODEL.predict_proba(__TRANSFORM_TEXT))))
                
        return {
            "sentiment" : __SENTIMENT,
            "confidence" : __SENTIMENT_CONFIDENCE
        }
        
    def getSentiment(self) -> str:
        #__preprocessed_text = self.preprocess() 
        
        __BASE_MODEL = BaseModel(
            text=self.__text, 
            languageData=self.__getLanguage(),
            sentimentData= self.__getSentiment(),
            textEmotionData = self.__getTextEmotion()
            )
        return __BASE_MODEL.to_json()
        
    def getMultipleSentiment(self) -> dict:
        __ALL_SENTIMENT = {
            "sentiments":[]
        }
        for text in self.__textList:
            self.__text = text
            __ALL_SENTIMENT["sentiments"].append(self.getSentiment())
        
        return __ALL_SENTIMENT
