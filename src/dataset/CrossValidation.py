__all__ = ["CrossValidator"]

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit
)

from src.util.util_function import SEED, ROOT_PATH


class CrossValidator:
    def __init__(self, cv_type="holdout", n_splits=5, shuffle=True, test_size=0.2, random_state=SEED):
        """
        다양한 Cross Validation 기법을 활용하여 데이터 분할만 수행하는 클래스
        - cv_type: "holdout", "kfold", "stratified", "timeseries" 선택 가능
        - n_splits: K-Fold, Stratified K-Fold, TimeSeriesSplit에서 사용할 폴드 개수
        - test_size: Hold-out 방식에서 테스트 데이터 비율
        - shuffle: 데이터 섞기 여부
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.test_size = test_size
        self.random_state = random_state
        self.cv = self._initialize_cv()


    def _initialize_cv(self):
        """ 선택한 교차 검증 방식에 맞는 객체 초기화 """
        if self.cv_type == "holdout":
            return None  # Hold-out은 별도로 train_test_split 사용
        elif self.cv_type == "kfold":
            return KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        elif self.cv_type == "stratified":
            # target이 연속형 변수일 경우 StratifiedKFold는 사용할 수 없음
            # 연속형 변수일 경우 KFold를 사용하도록 설정
            if isinstance(self.n_splits, int) and self.n_splits > 1:
                return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            else:
                raise ValueError("StratifiedKFold는 n_splits가 2 이상이어야 합니다.")
        elif self.cv_type == "timeseries":
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError("지원되지 않는 Cross Validation 방식입니다.")
        

    def split(self, X, y=None):
        """ 데이터를 선택한 방식으로 분할하여 반환 """

        splits = []
        if self.cv_type == "holdout":
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            
            return X_train, X_val, y_train, y_val
        else:
            splits = []
            # x 데이터는 2차원으로 들어오는데, y 데이터는 1차원으로 들어온다.
            for train_index, val_index in self.cv.split(X, y):
                X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                splits.append((X_train, X_val, y_train, y_val))

            return splits