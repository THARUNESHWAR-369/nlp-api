
import pickle
import os

class LOAD_MODEL:
    
    __DIR = "models"
    
    def __init__(self, base_type: str, version: str = "v1", model_type: str = "ml-models") -> None:
        
        self.__MODEL_TYPE = model_type
        
        self.__BASE_TYPE = base_type
        
        self.__VERSION = version
        
        self.__MODEL_DIR = f"{self.__DIR}/{self.__MODEL_TYPE}/{self.__BASE_TYPE}/{self.__VERSION}"
        
    
    def __get_subDirs(self) -> list:
        __PATHS = []
        
        for __sd in os.listdir(self.__MODEL_DIR):
            __PATHS.append(f"{self.__MODEL_DIR}/{__sd}")
            
        return __PATHS
    
    def __get_modelPath(self, paths) -> list:
        
        __MODEL_PATHS = []
        for path in paths:
            __MODEL_PATHS.append(f"{path}/{os.listdir(path)[0]}")
        return __MODEL_PATHS
    
    def load(self) -> list:
        __PATHS = self.__get_subDirs()
        __MODEL_PATHS = self.__get_modelPath(__PATHS)
        
        __MODELS = []
        
        for models in __MODEL_PATHS:
            
            print(models)
            
            __MODELS.append(pickle.load(open(models, 'rb')))
            
        return __MODELS
        
        
    