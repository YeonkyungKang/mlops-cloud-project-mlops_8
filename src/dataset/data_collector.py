from datetime import datetime, timedelta
import requests
import pandas as pd
import fire
from dotenv import load_dotenv
import os
import sys

from pathlib import Path
import shutil

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# .env 파일 로드
load_dotenv()

KMA_API_KEY = os.getenv("KMA_API_KEY")
current_date = datetime.now().strftime('%Y%m%d')
last_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
temp_dir = os.path.join(ROOT_DIR, 'temp')
os.makedirs(temp_dir, exist_ok=True)

seoul_tour_hotspot = pd.read_csv(f'{ROOT_DIR}/dataset/seoul_tour_hotspot.csv')

def get_date_ranges(start_date_str):
    def _get_month_end_date(date):
        """Helper function to get last day of month"""
        if date.month == 12:
            next_month = date.replace(year=date.year+1, month=1, day=1)
        else:
            next_month = date.replace(month=date.month+1, day=1)
        return next_month - pd.Timedelta(days=1)
    
    def _increment_month(date):
        if date.month == 12:
            return date.replace(year=date.year+1, month=1)
        return date.replace(month=date.month+1)


    if start_date_str == last_date:
        return [(last_date, last_date)]
    
    current_date = datetime.now()
    current_date_str = current_date.strftime('%Y%m%d')

    start_date = datetime.strptime(start_date_str, '%Y%m%d').replace(day=1)
    date_ranges = []
    date = start_date

    while date <= current_date:
        start_day = date.replace(day=1)
        end_day = _get_month_end_date(date)
        
        if end_day > current_date:
            end_day = current_date
            
        date_range = (start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d'))
        date_ranges.append(date_range)
        
        date = _increment_month(date)

    return date_ranges

def get_base_url(url_type="temperature"):
    url_tag = ''
    if url_type == 'temperature':       # 기후
        url_tag = 'sts_ta'
    elif url_type == 'insolation':   # 일사
        url_tag = 'sts_si'   
    elif url_type == 'wind':          # 바람
        url_tag = 'sts_wind'
    elif url_type == 'pressure':      # 기압
        url_tag = 'sts_pa'
    elif url_type == 'humidity':      # 습도
        url_tag = 'sts_rhm'
    elif url_type == 'rainfall': # 강수량
        url_tag = 'sts_rn'

    return str(f"https://apihub.kma.go.kr/api/typ01/url/{url_tag}.php")


def get_weather_data(date_ranges, url_type='temperature'):
    print(f"Downloading {url_type} data...")
    base_url = get_base_url(url_type)
    all_responses = []
    for idx, row in seoul_tour_hotspot.iterrows():
        lat = row['위도(도)']
        lon = row['경도(도)']
        spot_id = row['관광지 아이디']
        for start, end in date_ranges:
            params = {
                'tm1': start,
                'tm2': end, 
                'lat': str(lat),  # Latitude parameter
                'lon': str(lon),  # Longitude parameter
                'disp': 0,
                'authKey': KMA_API_KEY
            }
            
            response = requests.get(base_url, params=params)
            response.spot_id = spot_id
            all_responses.append(response)
            response = all_responses[-1] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_file_path = f'{temp_dir}/{url_type}_data_{timestamp}.txt'
        
    # Write responses to file
    head_tag = False
    with open(save_file_path, 'w', encoding='utf-8') as f:
        for resp in all_responses:
            if resp.status_code == 200: 
                response_data = resp.text.strip()
                if response_data:
                    lines = response_data.split('\n')
                    if head_tag == False: head_tag = True
                    else: lines = lines[1:]

                    modified_lines = [f"{resp.spot_id},{line.replace(chr(9),',')}" for line in lines]
                    f.write('\n'.join(modified_lines))
                    f.write('\n')  
            else:
                print(f"Error occurred while fetching data for spot_id {resp.spot_id}:")
                print(f"Status code: {resp.status_code}")
                print(f"Error message: {resp.text}")
                print(f"Request URL: {resp.url}")
    print(f"Downloaded {url_type} data to {save_file_path}")
    return save_file_path


def data_merge(temperature_file, wind_file, humidity_file, rainfall_file):
    temperature_data = pd.read_csv(temperature_file, skiprows=1, header=None)
    wind_data = pd.read_csv(wind_file, skiprows=1, header=None)
    humidity_data = pd.read_csv(humidity_file, skiprows=1, header=None)
    rainfall_data = pd.read_csv(rainfall_file, skiprows=1, header=None)

    tiny_temperature_data = temperature_data[[0, 1, 2, 3, 4, 7]]
    tiny_temperature_data.columns = ['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON', 'Average_temperature']
  
    tiny_wind_data = wind_data[[0, 1, 2, 3, 4, 6]]
    tiny_wind_data.columns = ['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON', 'Average_wind_speed']

    tiny_humidity_data = humidity_data[[0, 1, 2, 3, 4, 7]]
    tiny_humidity_data.columns = ['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON', 'Average_humidity']

    tiny_rainfull_data = rainfall_data[[0, 1, 2, 3, 4, 6, 7, 8]]
    tiny_rainfull_data.columns = ['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON', 'Sum_rainfall', 'Max_rainfall_1H', 'Max_rainfall_1H_occur_time']

    weather_data = tiny_temperature_data.merge(tiny_rainfull_data, on=['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON'], how='left')
    weather_data = weather_data.merge(tiny_humidity_data, on=['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON'], how='left')
    weather_data = weather_data.merge(tiny_wind_data, on=['Spot_id', 'YMD', 'STN_ID', 'LAT', 'LON'], how='left')

    filename = f'{ROOT_DIR}/dataset/weather_data_{current_date}.csv'
    weather_data.to_csv(filename, index=False)
    print(f"Merged data saved to {filename}")

    return filename

def all_data_download(start_dat):

    date_ranges = get_date_ranges(start_dat)
    print(f"Date Range: {date_ranges[0][0]} to {date_ranges[-1][1]}")
    temperature_file = get_weather_data(date_ranges, 'temperature')
    #insolation_file = get_weather_data(date_ranges, 'insolation')
    wind_file = get_weather_data(date_ranges, 'wind')
    #pressure_file = get_weather_data(date_ranges, 'pressure')
    humidity_file = get_weather_data(date_ranges, 'humidity')
    rainfall_file = get_weather_data(date_ranges, 'rainfall')

    merge_file = data_merge(temperature_file, wind_file, humidity_file, rainfall_file)

    return merge_file

def show_help():
    print("Usage:")
    print("  python data_collector.py init      # 전체 데이터 다운로드")
    print("  python data_collector.py update    # 어제 데이터 다운로드")



if __name__ == "__main__":

    fire.Fire({
        "init": all_data_download('20230401'),
        "update": all_data_download(last_date),
        "__default__": show_help,
    })
