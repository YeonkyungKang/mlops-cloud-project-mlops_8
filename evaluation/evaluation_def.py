#!pip install plotly #:시작 시 이 코드 반드시 실행하고 다시 주석처리 해야 함

'''
#한글 깨짐 방지 코드
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'//home/yes//NANUMBARUNGOTHIC.TTF', 
    ## 나눔바른고딕 글자 설정 -> ttf 파일 없거나 + 에러 나면 한글 깨짐 방지 코드 전체 삭제해도 괜찮음
    name='NanumBarunGothic')
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'})
plt.rc('font', family='NanumBarunGothic')
''' 



## 1. model, X_train 기반 feature importance 측정하는 함수

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importances(model, X_train, top_n=None, figsize=(10,8), title="Feature Importances"):
    """
    트리 기반 머신러닝 모델의 feature importance를 시각화합니다.

    Parameters:
        model: 학습된 트리 기반 모델 (feature_importances_ 속성 필요)
        X_train: 학습 데이터프레임 (특성 이름 추출용)
        top_n: 상위 n개 특성만 시각화 (None이면 전체)
        figsize: 플롯 크기 (기본값 (10,8))
        title: 그래프 제목
    """
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    if top_n is not None:
        importances = importances.head(top_n)
    plt.figure(figsize=figsize)
    plt.title(title)
    sns.barplot(x=importances, y=importances.index)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# 실제 사용법: plot_feature_importances(model, X_train, top_n=5) #top_n 숫자는 사용자가 임의로 설정


## 2. model, X_val,y_val 기반 feature importance 측정하는 함수 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

def plot_permutation_importance(model, X_val, y_val, n=20, order='high'):
    """
    permutation importance를 계산하고, 상위 또는 하위 n개의 feature importance를 plotly로 시각화하며,
    해당 feature importance 정보를 DataFrame으로 반환하는 함수입니다.

    Parameters:
    - model: 학습된 모델
    - X_val: 검증용 feature DataFrame
    - y_val: 검증용 target
    - n: 시각화할 feature 개수 (상위 또는 하위)
    - order: 'high' 또는 'low' 선택, 상위 또는 하위 feature importance 시각화

    Returns:
    - importance_df: 선택된 feature importance 정보가 담긴 DataFrame
    """
    # permutation importance 계산
    perm_result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring='neg_root_mean_squared_error',
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importances = perm_result.importances_mean
    std = perm_result.importances_std
    features = X_val.columns

    # order에 따라 상위 또는 하위 n개 선택
    if order == 'high':
        indices = np.argsort(importances)[::-1][:n]
    elif order == 'low':
        indices = np.argsort(importances)[:n]
    else:
        raise ValueError("order는 'high' 또는 'low'만 가능합니다.")

    selected_features = features[indices]
    selected_importances = importances[indices]
    selected_std = std[indices]

    # DataFrame 생성
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': selected_importances,
        'Std': selected_std
    })

    # plotly 시각화
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=selected_features,
        y=selected_importances,
        error_y=dict(type='data', array=selected_std),
        text=[f'{imp:.4f}' for imp in selected_importances],
        hoverinfo='text',
        marker_color='royalblue'
    ))

    fig.update_layout(
        title=f"Permutation Feature Importance ({order.capitalize()} {n} Features)",
        xaxis_title="Feature",
        yaxis_title="Importance",
        xaxis_tickangle=-45,
        template='plotly_white'
    )

    fig.show()

    return importance_df

'''
# plot_permutation_importance(model, X_val, y_val, n=5, order='high')
# 옵션 사용 설명서 
# model, X_val,y_val 입력
# order 옵션: 'high' ,'low' : feature importance 상위, 하위 설정
# n 옵션: feature 몇 개까지 출력할지

# 사용 예시
# plot_permutation_importance(model, X_val, y_val, n=5, order='high') -> feature importance 상위 5개까지 출력
# plot_permutation_importance(model, X_val, y_val, n=4, order='low') -> feature importance 하위 4개까지 출력
'''






## 3. 주요 통계 지표 확인

#주요 통계 지표 확인 및 예측 값과 실제 값의 분포가 잘 되었는지 그래프, 지표값으로 확인하는 함수

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_statistics(pred_original, y_val_original):
    """
    pred_original과 y_val_original의 주요 통계 지표(평균, 표준편차, 중앙값, 최소, 최대 등)를 계산하여 DataFrame으로 반환
    """
    stats_pred = {
        'mean': np.mean(pred_original),
        'std': np.std(pred_original),
        'median': np.median(pred_original),
        'min': np.min(pred_original),
        'max': np.max(pred_original)
    }
    stats_y_val = {
        'mean': np.mean(y_val_original),
        'std': np.std(y_val_original),
        'median': np.median(y_val_original),
        'min': np.min(y_val_original),
        'max': np.max(y_val_original)
    }
    stats_df = pd.DataFrame({'pred_original': stats_pred, 'y_val_original': stats_y_val})
    return stats_df

def plot_distributions(pred_original, y_val_original, figsize=(12, 5)):
    """
    pred_original과 y_val_original 각각의 분포를 히스토그램과 커널 밀도 추정(KDE) 그래프로 시각화
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.histplot(pred_original, kde=True, color='blue')
    plt.title('Distribution of pred_original')
    plt.xlabel('pred_original')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(y_val_original, kde=True, color='green')
    plt.title('Distribution of y_val_original')
    plt.xlabel('y_val_original')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def analyze_and_plot(pred_original, y_val_original):
    """
    pred_original과 y_val_original의 주요 통계 지표를 출력하고,
    각각의 분포를 시각화하는 함수
    """
    stats_df = calculate_statistics(pred_original, y_val_original)
    print("주요 통계 지표:")
    print(stats_df)
    plot_distributions(pred_original, y_val_original)
    return stats_df

# analyze_and_plot(pred_original, y_val_original)

# 함수에서 입력 인자로 y_val_original, pred_original은 모델 학습 완료 후 로그 역변환 완료 후의 y_val, pred 값을 말함 
# analyze_and_plot(pred_original, y_val_original)
# 출력값은 지표 나오는 df와 plotly로 나오는 그래프




## 4. 주요 성능 평가 지표 구하는 함수
# 성능 지표: MAE, MAPE, RMSE, 상관계수, R^2

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

def regression_metrics(y_val_original, pred_original):
    """
    y_val_original(실제값), pred_original(예측값)을 입력받아
    MAE, MAPE, RMSE, 상관계수, R^2를 계산하여 출력하는 함수

    Parameters:
    - y_val_original: 실제값 (array-like)
    - pred_original: 예측값 (array-like)

    Returns:
    - metrics_dict: 각 지표별 결과가 담긴 딕셔너리
    """
    y = np.array(y_val_original)
    p = np.array(pred_original)
    
    mae = mean_absolute_error(y, p)
    mape = np.mean(np.abs((y - p) / (y + 1e-8))) * 100  # 0 나누기 방지 #mape 단위는 % 
    #rmse = mean_squared_error(y, p, squared=False)
    rmse = mean_squared_error(y, p) ** 0.5
    corr = np.corrcoef(y, p)[0, 1]
    r2 = r2_score(y, p)
    
    metrics_dict = {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'Correlation': corr,
        'R2': r2
    }
    
    # 결과 출력
    print("회귀 성능 지표:")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    return metrics_dict


# regression_metrics(y_val_original, pred_original)

# 함수에서 입력 인자로 y_val_original, pred_original은 모델 학습 완료 후 로그 역변환 완료 후의 y_val, pred 값을 말함 




