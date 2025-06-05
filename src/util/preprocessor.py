import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class WeatherPreprocessor:
    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self.feature_scaler = None  # Feature scaling
        self.target_scaler = None   # Target scaling
        self.label_encoders = {}

    def fill_missing(self, df: pd.DataFrame, method: str = "ffill"):
        """결측치 처리"""
        if method == "ffill":
            return df.fillna(method="ffill").fillna(method="bfill")
        elif method == "zero":
            return df.fillna(0)
        else:
            return df.dropna()

    def encode_categorical(self, df: pd.DataFrame, columns: list):
        """범주형 변수 인코딩"""
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def scale_numeric_features(self, df: pd.DataFrame, columns: list):
        """수치형 피처 변수 스케일링 (타겟 제외)"""
        if not columns:
            return df
        if self.scaler_type == "standard":
            self.feature_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
        df[columns] = self.feature_scaler.fit_transform(df[columns])
        return df

    def scale_target(self, df: pd.DataFrame, target_column: str):
        """타겟 변수 스케일링"""
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found for scaling.")
            return df
        
        if self.scaler_type == "standard":
            self.target_scaler = StandardScaler()
        else:
            self.target_scaler = MinMaxScaler()
        
        df[target_column] = self.target_scaler.fit_transform(df[[target_column]])
        return df
    
    def log_transform(self, df, columns):
        """로그 변환"""
        for col in columns:
            df[f'{col}_log'] = np.log1p(df[col])
        return df
    
    def detect_outliers(self, df, columns):
        """이상치 확인"""
        outlier_info = {}
        
        # 박스플롯 그리기
        plt.figure(figsize=(15, 5 * ((len(columns) + 1) // 2)))
        for idx, column in enumerate(columns, 1):
            plt.subplot((len(columns) + 1) // 2, 2, idx)
            sns.boxplot(x=df[column])
            plt.title(f'{column} 분포')
            
            # IQR 방식으로 이상치 탐지
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)][column]
            
            outlier_info[column] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'min': df[column].min(),
                'max': df[column].max(),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
        
        plt.tight_layout()
        plt.show()
        
        # 이상치 정보 출력
        print("\n=== 이상치 분석 결과 ===")
        for column, info in outlier_info.items():
            print(f"\n[{column}]")
            print(f"이상치 개수: {info['outlier_count']}개")
            print(f"이상치 비율: {info['outlier_percentage']:.2f}%")
            print(f"데이터 범위: {info['min']:.2f} ~ {info['max']:.2f}")
            print(f"이상치 기준: {info['Q1'] - 1.5 * info['IQR']:.2f} ~ {info['Q3'] + 1.5 * info['IQR']:.2f}")

    # IQR 이상치 제거 함수
    def remove_outliers_iqr(self, dt, columns):
        """IQR 방식으로 이상치 제거"""
        dt = dt.copy()
        
        # float 타입 컬럼만 선택
        float_columns = [col for col in columns if dt[col].dtype in ['float64', 'float32']]
        
        for column in float_columns:
            Q1 = dt[column].quantile(0.25)
            Q3 = dt[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # 더 관대한 범위 설정 (1.5 -> 3.0)
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            # 이상치 처리 전후 통계 출력
            before_stats = dt[column].describe()
            
            dt.loc[dt[column] < lower_bound, column] = lower_bound
            dt.loc[dt[column] > upper_bound, column] = upper_bound
            
            after_stats = dt[column].describe()
            
            print(f"\n{column} 컬럼 처리 결과:")
            print("처리 전:", before_stats[['min', 'max', 'mean', 'std']].round(2))
            print("처리 후:", after_stats[['min', 'max', 'mean', 'std']].round(2))
        
        return dt

    def transform_datetime(self, df: pd.DataFrame, date_column: str):
        df['year'] = df[date_column].astype(str).str[:4].astype(int)
        df['month'] = df[date_column].astype(str).str[4:6].astype(int)
        df['day'] = df[date_column].astype(str).str[6:8].astype(int)

        return df

    @classmethod
    def inverse_transform_target(cls, target_scaler: StandardScaler | MinMaxScaler, target_scaled_values: np.ndarray) -> np.ndarray:
        """스케일링된 타겟 변수 역변환"""
        if target_scaler is None:
            raise ValueError("Target scaler object must be provided.")
        # Scaler expects 2D input for transform and inverse_transform
        if target_scaled_values.ndim == 1:
            # input 차원이 1차원 인 경우에
            # [
            #   [20],
            #   [10],   
            #]
            # 위와 같은 형태로 2차원 형태로 넣어줘야 하기 때문에 나중에 밖에서 flatten() 매서드로 반드시 1차원으로 만들어줄 것!!
            target_scaled_values = target_scaled_values.reshape(-1, 1)
        return target_scaler.inverse_transform(target_scaled_values)