__all__ = ['LightGBMTrainer']
import lightgbm as lgb
from lightgbm import Booster
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd # X_train.columns 사용 시 필요 (fit 메소드 내)

from .BaseTrainer import BaseTrainer
from src.util.util_function import SEED


class LightGBMTrainer(BaseTrainer):
    trainer_type_name = "lightgbm" # MLflowManager가 타입을 식별하기 위함

    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50,
                 log_period=0, categorical=None, verbose=True, log_transformed_target=True):
        """
        LightGBM 학습을 위한 클래스
        - params: LightGBM 하이퍼파라미터 딕셔너리
        - num_boost_round: 부스팅 라운드 횟수
        - early_stopping_rounds: 조기 종료 기준
        - log_period: 로그 기록 주기 (0이면 lgb.log_evaluation 콜백에 의해 로그가 출력되지 않음, lgb.train의 기본 동작으로 여전히 로그가 나올 수 있음)
        - categorical: 범주형 변수 리스트
        - verbose: 학습 과정 중 early stopping 메시지 등 일부 메시지 출력 여부
        - log_transformed_target: 타겟 변수가 로그 변환되었는지 여부 (True이면 predict, evaluate 시 np.expm1 적용)
        """
        self._model_params = params if params else self.default_params() # 모델 하이퍼파라미터용

        # 트레이너 자체 및 lgb.train에 직접 전달되는 설정값들
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds_config = early_stopping_rounds # 콜백 설정용
        self.log_period_config = log_period             # 콜백 설정용
        self.categorical_config = categorical         # lgb.Dataset 생성용
        self.verbose_trainer = verbose                  # 트레이너 자체 및 콜백의 verbose에 사용
        self.log_transformed_target = log_transformed_target

        self.evals_result = {}
        self.model: Booster = None
        self.features_importance = None # LightGBM은 NumPy 배열을 반환
        self.features_name = None       # LightGBM은 리스트를 반환

        self.setting_callback_function() # __init__에서 콜백 설정

    def default_params(self):
        """ 기본 LightGBM 하이퍼파라미터 설정 """
        return {
            'seed': SEED,
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'num_leaves': 80,
            'boosting_type': 'gbdt',
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'feature_fraction': 0.8,
        }

    def get_model_hyperparams(self):
        """lgb.train 함수의 params 인자로 전달될 핵심 하이퍼파라미터들을 반환합니다."""
        return self._model_params.copy() # 내부 변경 방지를 위해 복사본 반환

    def get_trainer_configs(self):
        """
        트레이너 자체의 주요 설정값들 (모델 하이퍼파라미터 제외)을 반환합니다.
        """
        return {
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds_config,
            'log_period': self.log_period_config,
            'categorical_features': self.categorical_config,
            'verbose_trainer': self.verbose_trainer,
            'log_transformed_target': self.log_transformed_target
        }

    def setting_callback_function(self):
        """ 콜백 함수 설정 """
        self.callbacks = []
        if self.early_stopping_rounds_config > 0: # __init__에서 설정된 _config 값 사용
            self.callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds_config, verbose=self.verbose_trainer)
            )

        # self.log_period_config > 0 일 때만 lgb.log_evaluation 추가 (0이면 추가 안함)
        # lgb.log_evaluation의 period가 음수이면 1로 처리됨 (LightGBM 동작)
        if self.log_period_config != 0: # 0이 아닐 때만 추가 (양수 또는 음수)
            self.callbacks.append(lgb.log_evaluation(period=self.log_period_config if self.log_period_config > 0 else 1))

        self.callbacks.append(lgb.record_evaluation(eval_result=self.evals_result))

    def save_model_info(self):
        """ 모델 학습 후 특성 중요도 및 특성 이름 저장 """
        if self.model:
            self.features_importance = self.model.feature_importance(importance_type="gain")
            self.features_name = self.model.feature_name()
        elif self.verbose_trainer:
            print("모델이 학습되지 않아 특성 정보를 저장할 수 없습니다.")

    def fit(self, X_train, y_train, X_val, y_val):
        """ 모델 학습 """
        # --- MLflow 관련 코드 없음 ---
        # lgb.Dataset 생성 시 categorical_feature 파라미터에 self.categorical_config 사용
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.categorical_config)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=self.categorical_config)

        current_model_params = self.get_model_hyperparams()

        if self.verbose_trainer:
            # lgb.train 자체의 verbose_eval은 콜백으로 제어되므로, 여기서는 트레이너 시작 메시지만 출력
            # 실제 로그 빈도는 setting_callback_function의 lgb.log_evaluation 콜백과 log_period_config에 의해 결정됨
            # 또는 MLflowManager 사용 시 autolog가 제어할 수 있음
            print(f"Starting LightGBM training...")
            # print(f"Model Hyperparameters: {current_model_params}") # 필요시 확인용

        # lgb.train 호출 시 verbose_eval 파라미터는 사용하지 않음 (콜백으로 제어)
        # LightGBM은 verbose_eval=-1 (또는 params에 'verbose': -1)로 설정하면 C++ 레벨 로그 줄임
        # 콜백을 통한 로그(log_evaluation)는 여기서 별도 제어
        self.model = lgb.train(
            current_model_params, # 모델 하이퍼파라미터
            train_data,
            num_boost_round=self.num_boost_round, # 트레이너 설정값
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=self.callbacks # __init__에서 설정된 콜백
        )
        self.save_model_info()
        if self.verbose_trainer:
            print("LightGBM training finished.")
        return self

    def predict(self, X_test):
        """ 예측 수행 """
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요! (Model is not trained yet.)")
        predictions = self.model.predict(X_test)
        return np.expm1(predictions) if self.log_transformed_target else predictions

    def evaluate(self, X_test, y_test):
        """ RMSE 평가 수행 """
        # --- MLflow 관련 코드 없음 ---
        y_pred = self.predict(X_test)
        y_test_original = np.expm1(y_test) if self.log_transformed_target else y_test

        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

        # 평가지표 이름 가져오기 (출력용)
        eval_metric_param = self.get_model_hyperparams().get('metric', 'rmse')
        if isinstance(eval_metric_param, list): # metric이 리스트일 수 있음
            metric_name_to_print = eval_metric_param[0]
        else:
            metric_name_to_print = eval_metric_param

        if self.verbose_trainer:
            print(f'Test 데이터 {metric_name_to_print.upper()}: {rmse:.4f}')
        return rmse