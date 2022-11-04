import pickle

from .text_preprocessor import NLP_PREPROCESS

from modules.load_model import LOAD_MODEL




class SENTIMENT_ANALYSIS(NLP_PREPROCESS):
    
    def __init__(self, text: str = None, textList: list = None, languageDetectionModelVersion: str = 'v2', sentimentAnalysisModelVersion: str = 'v1') -> None:
        super().__init__(text)
        self.__text = text
        self.__textList = textList
        self.__language_detection_model_version = languageDetectionModelVersion
        self.__sentiment_analysis_model_version = sentimentAnalysisModelVersion
                
        self.__loadModel()
        
    def __mapToString(self, prediction) -> str:
        if prediction == 0:
            return "NEGATIVE"
        elif prediction == 1:
            return "NEUTRAL"
        else:
            return "POSITIVE"
        
    def __loadModel(self) -> None:
        self.__MODEL, self.__VECTORIZER = LOAD_MODEL(base_type="sentiment-analysis-model", version=self.__sentiment_analysis_model_version).load()
        
    def __getLanguage(self) -> str:
        __LABEL_ENCODER, __MODEL, __VECTORIZER = LOAD_MODEL(base_type="language-detection-model", version=self.__language_detection_model_version).load()
    
        
        return str(__LABEL_ENCODER.inverse_transform(__MODEL.predict(__VECTORIZER.transform([self.__text]).toarray()))[0]).upper()
        
    def getSentiment(self) -> str:
        __preprocessed_text = self.preprocess()
        return {
            "text": self.__text,
            #"preprocessed-text": __preprocessed_text  ,
            "language": self.__getLanguage(),
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
