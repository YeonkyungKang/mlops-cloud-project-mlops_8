__all__ = ['XGBoostTrainer']
import xgboost as xgb
from xgboost import Booster
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd # X_train.columns 사용 시 필요

from .BaseTrainer import BaseTrainer
from util.util_function import SEED


class XGBoostTrainer(BaseTrainer):
    trainer_type_name = "xgboost" # MLflowManager가 타입을 식별하기 위함

    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50,
                 eval_period=0, # XGBoost의 verbose_eval과 유사한 역할
                 categorical_features=None, # 참고용 (XGBoost는 params의 'enable_categorical' 등으로 처리)
                 verbose=True, log_transformed_target=True):
        """
        XGBoost 학습을 위한 클래스
        """
        self._model_params = params if params else self.default_params() # 모델 하이퍼파라미터용
        
        # 트레이너 자체 및 xgb.train에 직접 전달되는 설정값들
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds_config = early_stopping_rounds if early_stopping_rounds > 0 else None # 콜백 설정용
        self.eval_period = eval_period
        self.categorical_features = categorical_features
        self.verbose = verbose # 트레이너 자체 및 콜백의 verbose에 사용
        self.log_transformed_target = log_transformed_target

        self.evals_result = {}
        self.model: Booster = None
        self.feature_importances_ = np.array([])
        self.feature_names_ = []
        self.callbacks = [] # 콜백은 _setup_callbacks에서 설정됨

    def default_params(self):
        """ 기본 XGBoost 하이퍼파라미터 설정 """
        return {
            'seed': SEED,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            # 'tree_method': 'hist',
            # 'enable_categorical': True, # params 딕셔너리에 포함되어 get_model_hyperparams()로 전달됨
        }

    def get_model_hyperparams(self):
        """xgb.train 함수에 전달될 핵심 하이퍼파라미터들을 반환합니다."""
        return self._model_params.copy() # 내부 변경 방지를 위해 복사본 반환

    def get_trainer_configs(self):
        """
        트레이너 자체의 주요 설정값들 (모델 하이퍼파라미터 제외)을 반환합니다.
        MLflow 로깅 시 `trainer_` 접두사를 붙여 모델 하이퍼파라미터와 구분할 수 있습니다.
        """
        return {
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds_config, # 콜백 설정에 사용된 값
            'eval_period': self.eval_period, # xgb.train의 verbose_eval에 사용될 값
            'categorical_features': self.categorical_features, # DMatrix 생성 시 참고용
            'verbose_trainer': self.verbose, # 트레이너 및 콜백의 상세 로깅 여부
            'log_transformed_target': self.log_transformed_target
        }

    def _setup_callbacks(self):
        """ XGBoost 학습에 사용될 콜백 함수들을 설정합니다. """
        self.callbacks = []
        # self.early_stopping_rounds_config는 __init__에서 None 또는 양수로 설정됨
        if self.early_stopping_rounds_config is not None:
            eval_metric_param = self.get_model_hyperparams().get('eval_metric', 'rmse') # _model_params 사용
            if isinstance(eval_metric_param, list):
                metric_for_early_stopping = eval_metric_param[-1]
            elif isinstance(eval_metric_param, str):
                metric_for_early_stopping = eval_metric_param.split(',')[-1].strip()
            else:
                metric_for_early_stopping = eval_metric_param

            self.callbacks.append(xgb.callback.EarlyStopping(
                rounds=self.early_stopping_rounds_config,
                metric_name=metric_for_early_stopping,
                save_best=True,
                maximize=False,
            ))

        try:
            self.callbacks.append(xgb.callback.record_evaluation(self.evals_result))
        except AttributeError:
            if self.verbose:
                print("Warning: xgb.callback.record_evaluation not found. Eval history (self.evals_result) "
                      "might not be populated. Consider upgrading XGBoost to version >= 1.6.")

    def save_model_info(self):
        """ 모델 학습 후 특성 중요도 및 특성 이름 저장 """
        if self.model:
            raw_importances = self.model.get_score(importance_type='gain')
            if hasattr(self.model, 'feature_names') and self.model.feature_names:
                self.feature_names_ = self.model.feature_names
                self.feature_importances_ = np.array([raw_importances.get(f, 0.0)
                                                      for f in self.feature_names_])
            elif raw_importances:
                self.feature_names_ = list(raw_importances.keys())
                self.feature_importances_ = np.array([raw_importances[f]
                                                      for f in self.feature_names_])
            else:
                self.feature_names_ = []
                self.feature_importances_ = np.array([])
        elif self.verbose:
            print("모델이 학습되지 않아 특성 정보를 저장할 수 없습니다.")

    def fit(self, X_train, y_train, X_val, y_val):
        """ 모델 학습 """
        # --- MLflow 관련 코드 없음 ---
        train_feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        val_feature_names = list(X_val.columns) if isinstance(X_val, pd.DataFrame) else None
        
        # 모델 하이퍼파라미터 가져오기
        current_model_params = self.get_model_hyperparams()

        dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=train_feature_names,
                             enable_categorical=current_model_params.get('enable_categorical', False))
        dval = xgb.DMatrix(X_val, label=y_val,
                           feature_names=val_feature_names,
                           enable_categorical=current_model_params.get('enable_categorical', False))

        evals = [(dtrain, 'train'), (dval, 'val')]
        self._setup_callbacks()

        current_verbose_eval = self.eval_period if self.eval_period > 0 else False

        if self.verbose:
             print(f"Starting XGBoost training (console verbose_eval: {current_verbose_eval})...")
             # print(f"Model Hyperparameters: {current_model_params}") # 필요시 확인용


        self.model = xgb.train(
            current_model_params, # 모델 하이퍼파라미터
            dtrain,
            num_boost_round=self.num_boost_round, # 트레이너 설정값
            evals=evals,
            callbacks=self.callbacks,
            verbose_eval=current_verbose_eval # 트레이너 설정값
        )

        self.save_model_info()
        if self.verbose:
            print("XGBoost training finished.")
        return self

    def predict(self, X_test):
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요! (Model is not trained yet.)")

        test_feature_names = list(X_test.columns) if isinstance(X_test, pd.DataFrame) else None
        dtest_feature_names = self.model.feature_names if hasattr(self.model, 'feature_names') and self.model.feature_names else test_feature_names

        dtest = xgb.DMatrix(X_test, feature_names=dtest_feature_names,
                            enable_categorical=self.get_model_hyperparams().get('enable_categorical', False))
        
        predictions = self.model.predict(dtest)
        return np.expm1(predictions) if self.log_transformed_target else predictions

    def evaluate(self, X_test, y_test):
        # --- MLflow 관련 코드 없음 ---
        y_pred = self.predict(X_test)
        y_test_for_eval = np.expm1(y_test) if self.log_transformed_target else y_test
        
        rmse = np.sqrt(mean_squared_error(y_test_for_eval, y_pred))
        
        # 평가지표 이름 가져오기 (출력용)
        eval_metric_param = self.get_model_hyperparams().get('eval_metric', 'rmse')
        if isinstance(eval_metric_param, list):
            metric_name_to_print = eval_metric_param[0] # 첫번째 메트릭 또는 주 메트릭
        else:
            metric_name_to_print = eval_metric_param.split(',')[0].strip()

        if self.verbose:
            print(f'Test 데이터 {metric_name_to_print.upper()}: {rmse:.4f}')
        return rmse