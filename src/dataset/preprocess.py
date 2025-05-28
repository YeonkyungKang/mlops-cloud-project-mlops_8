import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from src.utils.utils import project_path
from src.utils.preprocessor import WeatherPreprocessor

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_weather_data(df):
    processor = WeatherPreprocessor(scaler_type="standard")  

    # 날짜 처리
    df = processor.transform_datetime(df, 'YMD')

    # 결측치 처리
    df = processor.fill_missing(df, method="zero")

    # 강수량이 -99.9 값인 경우 0으로 대체
    columns = ['Sum_rainfall', 'Max_rainfall_1H']
    for col in columns:
        df[col].replace(-99.9, 0, inplace=True)

    # 평균 기온, 평균 습도, 최저 습도가 -99.9 값인 경우 결측치로 대체하고 결측치를 앞행이나 뒷행의 값으로 대체
    columns = ['Average_temperature', 'Average_humidity', 'Min_humidity']    
    for col in columns:  
        df[col].replace(-99.9, np.nan, inplace=True)
    
    df = processor.fill_missing(df, method="ffill")    
    
    # 이상치 처리    
    # 수치형 컬럼만 선택, 이상치 제거
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df = processor.remove_outliers_iqr(df, numeric_columns)

    # 범주형 변수 인코딩
    # cat_columns = ['범주형 변수'] 
    # df = processor.encode_categorical(df, cat_columns)

    # 수치형 변수 스케일링
    num_columns = ['Average_temperature', 'Sum_rainfall', 'Max_rainfall_1H', 'Max_rainfall_1H_occur_time', 'Average_humidity', 'Min_humidity']  
    df = processor.scale_numeric(df, num_columns)

    # 로그 변환
    # log_columns = ['target'] # 로그 변환 변수 추가할 것
    # df = processor.log_transform(df, log_columns)


    # 칼럼 삭제
    columns_to_drop = ['Average_wind_speed'] # 삭제할 컬럼 있으면 추가할 것
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)    

    # 평균 기온을 타겟으로 설정
    df = df.rename(columns={'Average_temperature':'target'})

    return df

def validate_data(df):
    """데이터 유효성 검증"""
    assert not df.empty, "데이터프레임이 비어있습니다"
    
    # 필수 컬럼 존재 확인
    required_columns = [
        'YMD', 'Average_temperature', 'Sum_rainfall',
        'Average_humidity', 'Min_humidity'
    ]
    missing_cols = set(required_columns) - set(df.columns)
    assert not missing_cols, f"필수 컬럼이 누락됨: {missing_cols}"
    
    # 데이터 타입 검증
    numeric_cols = ['Average_temperature', 'Sum_rainfall', 'Average_humidity']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col}은 숫자형이어야 합니다"
    
    return True

if __name__ == "__main__":
    try:
        # 데이터 로드
        raw_data_path = os.path.join(project_path(), "dataset", "weather_data.csv")
        df = load_data(raw_data_path)
        
        # 데이터 검증
        validate_data(df)
        
        # 데이터 전처리
        processed_df = preprocess_weather_data(df)
        
        # 결과 저장
        output_path = os.path.join(project_path(), "dataset", "preprocessed_weather.csv")
        processed_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"전처리 완료: {len(processed_df)} 개의 데이터가 {output_path}에 저장되었습니다.")
        
    except Exception as e:
        print(f"전처리 중 오류 발생: {str(e)}")