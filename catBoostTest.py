import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from model.CatBoostTrainer import CatBoostTrainer  # CatBoostTrainer 클래스 임포트

# 데이터셋 로드 (회귀 문제)
data = fetch_california_housing()
X, y = data.data, data.target

# Pandas DataFrame으로 변환
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, y_val = X_test, y_test  # 간단한 테스트를 위해 test 데이터를 validation으로 재사용

# 기본 모델 설정
trainer = CatBoostTrainer()

def test_train():
    """ 모델 학습 테스트 """
    trainer.fit(X_train, y_train, X_val, y_val)
    assert trainer.model is not None, "모델이 정상적으로 학습되지 않음"

def test_predict():
    """ 예측 테스트 """
    trainer.fit(X_train, y_train, X_val, y_val)
    predictions = trainer.predict(X_test)
    assert isinstance(predictions, np.ndarray), "예측 결과가 numpy 배열이 아님"
    assert predictions.shape[0] == X_test.shape[0], "예측 결과의 개수가 입력 데이터 크기와 다름"

def test_evaluate():
    """ 평가 테스트 """
    trainer.fit(X_train, y_train, X_val, y_val)
    rmse = trainer.evaluate(X_test, y_test)
    assert rmse > 0, "RMSE 값이 0 이하일 수 없음"

# pytest 실행을 위한 코드
if __name__ == "__main__":
    pytest.main()