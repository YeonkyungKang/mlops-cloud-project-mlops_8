import pandas as pd

from src.model.LightGBMTrainer import LightGBMTrainer
from src.dataset.preprocess import get_datasets
from src.dataset.CrossValidation import CrossValidator


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
    rmse = trainer.evaluate(X_test, y_test)
    

if __name__ == "__main__":
    run_train()