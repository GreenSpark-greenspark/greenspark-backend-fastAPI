from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PowerData(BaseModel):
    date: str  # Spring에서 LocalDate 형태로 전달되므로 문자열로 수신
    cost: int

class predict_output(BaseModel):
    predicted_cost : float

@app.post("/ml", response_model=predict_output)
def predict_cost(input_data: List[PowerData]):
    # 데이터 로드
    file_path = '7data.csv'
    data = pd.read_csv(file_path)

    data['날짜'] = pd.to_datetime(data['날짜'], format='%b.%y', errors='coerce')
    data = data.rename(columns={'날짜': 'ds'})

    # 결측치 처리: 사용자별 결측치가 있는 경우, 평균 또는 중간값으로 대체
    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

    data.interpolate(method='linear', inplace=True)
    data.fillna(data.mean(), inplace=True)

    y_data = pd.DataFrame([{"ds": pd.to_datetime(item.date), "y": item.cost} for item in input_data])

    # print(y_data)

    # 기존 데이터와 H 사용자의 데이터를 병합
    combined_data = pd.concat([data, y_data], ignore_index=True)
    combined_data.interpolate(method='linear', inplace=True)
    combined_data.fillna(combined_data.mean(), inplace=True)

    # 하한값(floor) 및 상한값(cap) 추가
    current_max = combined_data['y'].max()
    combined_data['cap'] = current_max * 2
    combined_data['floor'] = 0

    # Prophet 모델 학습
    model = Prophet(
        yearly_seasonality=True,  # 연간 계절성 반영
        seasonality_mode='multiplicative',  # 곱셈 계절성 사용
        changepoint_prior_scale=0.1  # 변화점 민감도 조정
    )
    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        model.add_regressor(col)

    if not combined_data[['ds', 'y', 'A', 'B', 'C', 'D', 'E', 'F', 'G']].isnull().any().any():
        model.fit(combined_data[['ds', 'y', 'A', 'B', 'C', 'D', 'E', 'F', 'G']])

        # 미래 예측
        future = combined_data[['ds', 'A', 'B', 'C', 'D', 'E', 'F', 'G']].iloc[[-1]].copy()
        future['ds'] = future['ds'] + pd.DateOffset(months=1)
        future['cap'] = combined_data['cap'].iloc[-1]  
        future['floor'] = combined_data['floor'].iloc[-1]

        # 예측 수행
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))  # 후처리로 음수 제거
        # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        # print(forecast[['yhat']])

    # print(forecast)

    return {"predicted_cost": forecast['yhat'].values[0]}