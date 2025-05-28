__all__ = ['RandomForestTrainer']
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd # To check for DataFrame and get column names

from .BaseTrainer import BaseTrainer
from util.util_function import SEED

class RandomForestTrainer(BaseTrainer):
    trainer_type_name = "sklearn_randomforest" # MLflowManager가 타입을 식별하기 위함

    def __init__(self, params=None, n_estimators=100, # RandomForest의 주요 파라미터
                 trainer_verbose=True, log_transformed_target=True):
        """
        scikit-learn RandomForest 학습을 위한 클래스
        - params: RandomForestRegressor 하이퍼파라미터 딕셔너리 (n_estimators 제외)
        - n_estimators: 생성할 트리의 개수
        - trainer_verbose: 트레이너 자체의 일반 메시지 출력 여부
        - log_transformed_target: 타겟 변수가 로그 변환되었는지 여부
        """
        self._base_params = params if params else self.default_params() # n_estimators 제외한 기본 파라미터
        self._n_estimators = n_estimators # n_estimators는 별도로 받아 관리 후 모델 생성 시 주입

        self.trainer_verbose = trainer_verbose
        self.log_transformed_target = log_transformed_target

        self.model: RandomForestRegressor = None
        # RandomForest는 boosting 모델처럼 학습 중 eval set의 metric history를 직접 제공하지 않음
        self.evals_result_ = {} # 일관성을 위해 속성은 유지하되, 비어있을 수 있음
        self.feature_importances_ = np.array([])
        self.feature_names_ = []

    def default_params(self):
        """ 기본 RandomForestRegressor 하이퍼파라미터 설정 (n_estimators 제외) """
        return {
            'max_depth': None,          # 트리의 최대 깊이 (None이면 리프가 순수해지거나 min_samples_split에 도달할 때까지 확장)
            'min_samples_split': 2,     # 노드를 분할하기 위한 최소 샘플 수
            'min_samples_leaf': 1,      # 리프 노드가 되기 위한 최소 샘플 수
            'max_features': 1.0,     # 최적 분할을 찾을 때 고려할 특성 수 (1.0은 모든 특성)
            'random_state': SEED,       # 결과 재현성을 위한 시드
            'n_jobs': -1,               # 사용할 CPU 코어 수 (-1은 모든 코어 사용)
            # 'oob_score': False        # Out-of-bag 샘플을 사용하여 일반화 정확도 추정 여부
        }

    def get_model_hyperparams(self):
        """RandomForestRegressor 모델 생성자에 전달될 하이퍼파라미터를 반환합니다."""
        model_constructor_params = self._base_params.copy()
        model_constructor_params['n_estimators'] = self._n_estimators # __init__에서 받은 n_estimators 주입
        if 'random_state' not in model_constructor_params: # 기본값에 있지만 사용자가 params에서 제거했을 경우 대비
             model_constructor_params['random_state'] = SEED
        return model_constructor_params

    def get_trainer_configs(self):
        """트레이너 자체의 주요 설정값들을 반환합니다."""
        return {
            'n_estimators_config': self._n_estimators, # RandomForest의 핵심 설정
            'trainer_verbose_config': self.trainer_verbose,
            'log_transformed_target_config': self.log_transformed_target
            # RandomForest는 boosting 모델과 달리 early_stopping, log_period 등이 직접 적용되지 않음
        }

    def save_model_info(self):
        """ 모델 학습 후 특성 중요도 및 특성 이름 저장 """
        if self.model:
            # scikit-learn 0.24 이상에서 DataFrame으로 학습 시 feature_names_in_ 속성 사용 가능
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                self.feature_names_ = list(self.model.feature_names_in_)
            
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
            
            # feature_names_가 설정되지 않았고, 중요도 배열의 길이가 있다면 임시 이름 생성
            if not self.feature_names_ and len(self.feature_importances_) > 0:
                if len(self.feature_importances_) == self.model.n_features_in_: # n_features_in_은 sklearn 0.24+
                    self.feature_names_ = [f'feature_{i}' for i in range(self.model.n_features_in_)]
                else: # Fallback if n_features_in_ not available or mismatch
                    self.feature_names_ = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        elif self.trainer_verbose:
            print("모델이 학습되지 않아 특성 정보를 저장할 수 없습니다.")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """ 
        모델 학습. 
        RandomForestRegressor는 fit 시 X_val, y_val을 직접 사용하지 않습니다 (조기 종료 등).
        이 파라미터들은 다른 Trainer와의 인터페이스 일관성을 위해 유지됩니다.
        """
        # --- MLflow 관련 코드 없음 ---
        model_constructor_params = self.get_model_hyperparams()
        self.model = RandomForestRegressor(**model_constructor_params)

        if self.trainer_verbose:
            if X_val is not None or y_val is not None:
                print("Info: RandomForestTrainer's fit method received X_val/y_val, "
                      "but standard RandomForestRegressor does not use them during training "
                      "(e.g., for early stopping or periodic evaluation).")
            print(f"Starting RandomForest training with n_estimators={self.model.get_params()['n_estimators']}...")

        # scikit-learn 모델은 y_train이 1D 배열이어야 함 (필요시 .ravel() 등 사용)
        if hasattr(y_train, 'shape') and len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train_fit = y_train.ravel()
        else:
            y_train_fit = y_train
            
        self.model.fit(X_train, y_train_fit)

        # RandomForest는 fit 과정에서 evals_result_를 생성하지 않으므로 self.evals_result_는 비어있음
        # OOB 점수가 있다면 여기에 기록할 수 있으나, 현재는 기본 구조만 따름
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
             self.evals_result_['oob_score'] = self.model.oob_score_


        self.save_model_info()
        if self.trainer_verbose:
            print("RandomForest training finished.")
        return self

    def predict(self, X_test):
        """ 예측 수행 """
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요! (Model is not trained yet.)")
        
        predictions = self.model.predict(X_test)

        if self.log_transformed_target:
            return np.expm1(predictions) # 로그 변환된 예측값을 원래 스케일로 복원
        else:
            return predictions

    def evaluate(self, X_test, y_test):
        """ RMSE 평가 수행 """
        # --- MLflow 관련 코드 없음 ---
        y_pred = self.predict(X_test)

        y_test_for_eval = y_test
        if self.log_transformed_target:
            # y_test도 로그 변환된 상태로 전달되었다고 가정하고 원래 스케일로 복원
            y_test_for_eval = np.expm1(y_test)
            
        # 기본적으로 RMSE를 계산.
        metric_val = np.sqrt(mean_squared_error(y_test_for_eval, y_pred))
        metric_name_to_print = "RMSE" # 또는 get_model_hyperparams에서 metric 관련 정보 가져오기

        if self.trainer_verbose:
            print(f'Test 데이터 {metric_name_to_print}: {metric_val:.4f}')
        return metric_val