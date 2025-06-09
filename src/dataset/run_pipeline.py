# d:\Kaggle\mlops-cloud-project-mlops_8\src\dataset\run_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split

# Docker 이미지 내의 경로를 기준으로 import
# Dockerfile에서 WORKDIR /app 이고, util 폴더가 /app/util 로 복사됨
from preprocess import preprocess_weather_data, validate_data, split_test_train
from util.s3_handler import S3Handler
from util.util_function import SEED, ROOT_PATH


def main():
    print(f"Pipeline started. Using SEED: {SEED}")

    # 보통 원본 데이터를 저장하는 s3와 전처리된 데이터를 저장하는 s3 이렇게 두개로 아예 나눈다고 함.
    # # 환경 변수에서 S3 및 AWS 설정 로드
    # bucket_name = os.getenv("S3_BUCKET_NAME_ENV")
    # input_s3_key = os.getenv("S3_INPUT_KEY_ENV")
    # output_s3_key_train = os.getenv("S3_OUTPUT_KEY_TRAIN_ENV")
    # output_s3_key_test = os.getenv("S3_OUTPUT_KEY_TEST_ENV")
    # aws_access_key = os.getenv("MY_AWS_ACCESS_KEY_ENV")
    # aws_secret_key = os.getenv("MY_AWS_SECRET_KEY_ENV")
    # aws_region = os.getenv("MY_AWS_REGION_ENV", "ap-northeast-2")

    # required_env_vars = {
    #     "S3_BUCKET_NAME_ENV": bucket_name, "S3_INPUT_KEY_ENV": input_s3_key,
    #     "S3_OUTPUT_KEY_TRAIN_ENV": output_s3_key_train, "S3_OUTPUT_KEY_TEST_ENV": output_s3_key_test,
    #     "MY_AWS_ACCESS_KEY_ENV": aws_access_key, "MY_AWS_SECRET_KEY_ENV": aws_secret_key
    # }
    # if not all(required_env_vars.values()):
    #     missing_vars = [k for k, v in required_env_vars.items() if not v]
    #     print(f"오류: 필수 환경 변수가 누락되었습니다 - {missing_vars}")
    #     sys.exit(1)

    # 이 미니 프로젝트에서는 그냥 폴더를 나누는 형식으로 구분
    bucket_name_env = os.getenv("S3_BUCKET_NAME_ENV", "mlops-intelligence")
    access_key_env = os.getenv("MY_AWS_ACCESS_KEY_ENV",)
    secret_key_env = os.getenv("MY_AWS_SECRET_KEY_ENV",)
    region_env = os.getenv("MY_AWS_REGION_ENV", "ap-northeast-2")

    if not all([bucket_name_env, access_key_env, secret_key_env]):
        print("오류: S3_BUCKET_NAME_ENV, MY_AWS_ACCESS_KEY_ENV, MY_AWS_SECRET_KEY_ENV 환경 변수가 모두 설정되어야 합니다.")
        sys.exit(1)

    s3_handler = S3Handler(
        bucket_name=bucket_name_env,
        aws_access_key=access_key_env,
        aws_secret_key=secret_key_env,
        region=region_env
    )
    

    try:
        # 1. S3에서 원본 데이터 다운로드
        print(f"S3에서 데이터 다운로드 중: s3://{bucket_name_env}")
        df_raw = s3_handler.download_csv("original/weather_data_20250528.csv")
        print(f"원본 데이터 다운로드 완료. Shape: {df_raw.shape}")
        # 데이터 검증
        validate_data(df_raw)

        # 2. 데이터 전처리
        print("데이터 전처리 시작...")
        df_processed = preprocess_weather_data(df_raw)
        print(f"데이터 전처리 완료. Shape: {df_processed.shape}")

        # 3. 데이터 분할
        # 테스트 데이터가 인퍼런스 데이터일테니까 이 부분은 나누는걸 다른 방식으로 해야함
        train_df, test_df = split_test_train(df_processed)
        print(f"데이터 분할 완료. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # 4. 전처리된 학습 및 테스트 데이터를 S3에 업로드
        print(f"학습 데이터 업로드 중: s3://{bucket_name_env}")
        s3_handler.upload_csv(train_df, "preprocessed/preprocessed_weather_train.csv")
        
        print(f"테스트 데이터 업로드 중: s3://{bucket_name_env}")
        s3_handler.upload_csv(test_df, "preprocessed/preprocessed_weather_test.csv")
        
        print(f"스케일러 업로드 중: s3://{bucket_name_env}")
        scaler_local_path = os.path.join(ROOT_PATH, 'target_scaler.joblib')
        try:
            target_scaler_object = load(scaler_local_path) # 로컬에 저장된 스케일러 객체 로드
            s3_handler.upload_joblib(target_scaler_object, "preprocessed/target_scaler.joblib") # 객체를 S3에 업로드
        except FileNotFoundError:
            print(f"오류: 스케일러 파일 '{scaler_local_path}'을(를) 찾을 수 없습니다. S3에 업로드할 수 없습니다.")
        except Exception as e:
            print(f"오류: 스케일러 로드 또는 업로드 중 오류 발생 - {e}")

        print("파이프라인 성공적으로 완료.")

    except Exception as e:
        print(f"파이프라인 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
