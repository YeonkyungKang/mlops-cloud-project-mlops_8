__all__ = ['CatBoostTrainer']
from catboost import CatBoostRegressor # Pool, metrics는 직접 사용하지 않음 (Regressor 사용 시)
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd # To check for DataFrame and get column names

from .BaseTrainer import BaseTrainer
from util.util_function import SEED


class CatBoostTrainer(BaseTrainer):
    trainer_type_name = "catboost" # MLflowManager가 타입을 식별하기 위함

    def __init__(self, params=None, iterations=500, early_stopping_rounds=50,
                 log_output_period=0, # CatBoost의 fit 메소드 내 verbose 인자와 유사
                 cat_features_spec=None, # 범주형 특성 (인덱스 또는 이름 리스트)
                 trainer_verbose=True, # 트레이너 자체의 메시지 출력 여부
                 log_transformed_target=True):
        """
        CatBoost 학습을 위한 클래스
        """
        self._base_params = params if params else self.default_params() # 모델 하이퍼파라미터용 (iterations, early_stopping_rounds 제외)
        self._iterations = iterations # CatBoost 모델의 iterations 파라미터
        self._early_stopping_rounds_cb = early_stopping_rounds if early_stopping_rounds > 0 else None # CatBoost 모델의 early_stopping_rounds

        # 트레이너 자체 설정값들
        self.log_output_period = log_output_period
        self.cat_features_spec = cat_features_spec
        self.trainer_verbose = trainer_verbose
        self.log_transformed_target = log_transformed_target

        self.model: CatBoostRegressor = None
        self.evals_result_ = {}
        self.feature_importances_ = np.array([])
        self.feature_names_ = []

    def default_params(self):
        """ 기본 CatBoost 하이퍼파라미터 설정 (iterations, early_stopping_rounds 제외) """
        return {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            # 'learning_rate': 0.1, # CatBoost는 학습률 자동 튜닝 기능 사용 가능
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': SEED,
            'logging_level': 'Silent', # fit의 verbose로 주기적 출력 제어
        }

    def get_model_hyperparams(self):
        """CatBoost 모델 생성자(CatBoostRegressor)에 전달될 하이퍼파라미터를 반환합니다."""
        model_constructor_params = self._base_params.copy()
        model_constructor_params['iterations'] = self._iterations
        if self._early_stopping_rounds_cb is not None:
            model_constructor_params['early_stopping_rounds'] = self._early_stopping_rounds_cb
        return model_constructor_params

    def get_trainer_configs(self):
        """
        트레이너 자체의 주요 설정값들 (모델 하이퍼파라미터 제외)을 반환합니다.
        MLflow 로깅 시 `trainer_` 접두사를 붙여 모델 하이퍼파라미터와 구분할 수 있습니다.
        """
        return {
            'log_output_period': self.log_output_period,
            'cat_features_spec': self.cat_features_spec,
            'trainer_verbose': self.trainer_verbose,
            'log_transformed_target': self.log_transformed_target
            # _iterations, _early_stopping_rounds_cb는 get_model_hyperparams에 포함되므로 여기서는 제외하거나,
            # 필요시 'iterations_config' 와 같이 명시적으로 구분하여 추가할 수 있습니다.
            # 일관성을 위해 get_model_hyperparams()에 포함된 내용은 여기서 제외하는 것이 좋습니다.
        }

    def save_model_info(self):
        if self.model:
            if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
                self.feature_names_ = self.model.feature_names_
            self.feature_importances_ = self.model.get_feature_importance()
            if not self.feature_names_ and len(self.feature_importances_) > 0:
                self.feature_names_ = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        elif self.trainer_verbose:
            print("모델이 학습되지 않아 특성 정보를 저장할 수 없습니다.")

    def fit(self, X_train, y_train, X_val, y_val):
        # 모델 생성 시 사용할 실제 하이퍼파라미터 가져오기
        model_constructor_params = self.get_model_hyperparams()
        self.model = CatBoostRegressor(**model_constructor_params)

        fit_verbose_param = self.log_output_period if self.log_output_period > 0 else False

        if self.trainer_verbose:
            print(f"Starting CatBoost training (console verbose: {fit_verbose_param})...")
            # print(f"Model Hyperparameters: {model_constructor_params}") # 필요시 확인용

        current_cat_features = self.cat_features_spec
        # ... (cat_features 자동 감지 로직은 생략, 명시적 전달 권장) ...

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            cat_features=current_cat_features,
            verbose=fit_verbose_param
        )

        if hasattr(self.model, 'get_evals_result'):
            self.evals_result_ = self.model.get_evals_result()
        elif hasattr(self.model, 'evals_result_'):
             self.evals_result_ = self.model.evals_result_

        self.save_model_info()
        if self.trainer_verbose:
            print("CatBoost training finished.")
        return self

    def predict(self, X_test):
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요! (Model is not trained yet.)")
        predictions = self.model.predict(X_test)
        return np.expm1(predictions) if self.log_transformed_target else predictions

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test_for_eval = np.expm1(y_test) if self.log_transformed_target else y_test
        
        metric_val = np.sqrt(mean_squared_error(y_test_for_eval, y_pred))
        metric_name_to_print = self.get_model_hyperparams().get('eval_metric', 'RMSE')
        if isinstance(metric_name_to_print, list): # eval_metric이 리스트일 경우 첫번째 사용
            metric_name_to_print = metric_name_to_print[0]


        if self.trainer_verbose:
            print(f'Test 데이터 {metric_name_to_print}: {metric_val:.4f}')
        return metric_val