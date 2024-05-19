import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import urllib.request
from tqdm import tqdm
from transformers import BertTokenizer, TFBertForSequenceClassification
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import pickle
from prophet_model import prophet_model_predict

def simulate_trading(ticker, start_date, end_date_1, end_date_2,initial_capital,counter):
    counter = counter
    start_date = start_date
    end_date_1 = end_date_1
    end_date_2 = end_date_2

    finance_df,forecast = prophet_model_predict(ticker, start_date, end_date_1,counter)
    # 입력된 날짜 범위로 데이터 필터링
    filtered_df = finance_df[(finance_df['ds'] >= start_date) & (finance_df['ds'] <= end_date_1)]
    filtered_df_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date_2)]

    # 현재 자본 및 보유 주식 수 초기화
    current_capital = initial_capital
    shares_owned = 0

    # 현재 포지션 초기화
    position = 'None'

    # 실제 및 예측 주식 가격 설정
    actual_prices = filtered_df.set_index('ds')['y']
    predicted_prices = filtered_df_forecast.set_index('ds')['yhat']

    # 거래 기록 초기화
    trading_history = []

    # 시뮬레이션 시작
    for date, today_price, predicted_next_day_price in zip(actual_prices.index, actual_prices.values, predicted_prices.shift(-1).values):
        predicted_change = predicted_next_day_price - today_price

        # 매수 조건
        if predicted_change > 0 and position != 'Hold':
            shares_to_buy = current_capital / today_price
            shares_owned += shares_to_buy
            current_capital -= shares_to_buy * today_price
            position = 'Hold'
            trading_history.append('Buy')

        # 매도 조건
        elif predicted_change < 0 and position == 'Hold':
            current_capital += shares_owned * today_price
            shares_owned = 0
            position = 'None'
            trading_history.append('Sell')

        # 보유 조건
        elif position == 'Hold' and abs(predicted_change) <= 0.01 * today_price:
            trading_history.append('Hold')

        # 조치 없음 또는 보유
        else:
            if position == 'Hold':
                trading_history.append('Hold')
            else:
                trading_history.append('Stay')#조치없음

    # 최종 자본 및 수익률 계산

    # trading_history의 마지막 인덱스 반환
    last_record = trading_history[-1]
    return last_record
