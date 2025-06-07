import os

import boto3
import pandas as pd
from io import StringIO

class S3CSVHandler:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region="ap-northeast-2"):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.bucket_name = bucket_name

    def upload_csv(self, df, file_name):
        """DataFrame을 S3에 CSV 파일로 업로드"""
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=file_name,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        print(f"{file_name} 업로드 완료!")

    def download_csv(self, file_name):
        """S3에서 CSV 파일을 다운로드하여 DataFrame으로 반환"""
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_name)
        csv_data = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))
        print(f"{file_name} 다운로드 완료!")
        return df

# 사용 예시
if __name__ == "__main__":
    bucket_name_env = os.getenv("S3_BUCKET_NAME_ENV", "mlops-intelligence")
    access_key_env = os.getenv("MY_AWS_ACCESS_KEY_ENV")
    secret_key_env = os.getenv("MY_AWS_SECRET_KEY_ENV")
    region_env = os.getenv("MY_AWS_REGION_ENV", "ap-northeast-2")

    if not all([bucket_name_env, access_key_env, secret_key_env]):
        print("오류: S3_BUCKET_NAME_ENV, MY_AWS_ACCESS_KEY_ENV, MY_AWS_SECRET_KEY_ENV 환경 변수가 모두 설정되어야 합니다.")
        exit(1)

    s3_handler = S3CSVHandler(
        bucket_name=bucket_name_env,
        aws_access_key=access_key_env,
        aws_secret_key=secret_key_env,
        region=region_env
    )

    # CSV 파일 업로드
    sample_df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    s3_handler.upload_csv(sample_df, "sample.csv")

    # CSV 파일 다운로드
    downloaded_df = s3_handler.download_csv("sample.csv")
    print(downloaded_df)