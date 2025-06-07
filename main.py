import pandas as pd
import os
from joblib import load

from src.model.LightGBMTrainer import LightGBMTrainer
from src.dataset.preprocess import get_datasets
from src.dataset.CrossValidation import CrossValidator
from src.util.preprocessor import WeatherPreprocessor
from src.util.util_function import ROOT_PATH

def run_train():
    train_df, test_df = get_datasets()
    y = train_df["target"]
    X = train_df.drop(['target'], axis=1)
    y_test = test_df["target"]
    X_test = test_df.drop(['target'], axis=1)
    
    cv = CrossValidator()
    
    X_train, X_val, y_train, y_val = cv.split(X=X, y=y)
    
    trainer = LightGBMTrainer(num_boost_round=100, log_transformed_target=False)
    trainer.fit(X_train, y_train, X_val, y_val)
    
    predictions_scaled = trainer.model.predict(X_test)
    print(f"Scaled predictions length: {len(predictions_scaled)}")
    print(f"Scaled predictions (first 5): {predictions_scaled[:5]}")

    # 타겟 스케일러 로드 및 역변환
    scaler_path = os.path.join(ROOT_PATH, 'target_scaler.joblib')
    try:
        loaded_target_scaler = load(scaler_path) # 저장된 스케일러 객체 로드
        print(f"Target scaler loaded from {scaler_path}")

        # WeatherPreprocessor의 inverse_transform_target 클래스 메소드 사용
        predictions_original_scale = WeatherPreprocessor.inverse_transform_target(loaded_target_scaler, predictions_scaled)
        
        print(f"Original scale predictions (first 5): {predictions_original_scale.flatten()[:5]}")
    except FileNotFoundError:
        print(f"Error: Target scaler file not found at {scaler_path}. "
              "Ensure preprocessing script (get_datasets) was run and saved the scaler.")
    except Exception as e:
        print(f"Error during inverse transformation: {e}")

if __name__ == "__main__":
    run_train()