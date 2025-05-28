__all__ = ['BaseTrainer']
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """ 공통된 fit 함수를 가지는 추상 Trainer 클래스 """
    
    @abstractmethod
    def default_params(self) -> dict:
        """ 기본 하이퍼파라미터 설정 """
        return {}
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
    