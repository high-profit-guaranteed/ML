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

from bertcode import bertcode_predict

def prophet_model_predict(ticker, start_date, end_date_1, counter):
    counter = counter
    # 저장된 모델 파라미터를 불러오는 함수
    ticker = ticker
    start_date = start_date
    end_date_1 = end_date_1 
    finance_df,result_filtered_future = bertcode_predict(ticker,start_date,end_date_1,counter)
    def load_prophet_model_parameters(file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        return model_params
        
    file_path = f"prophet_model_parameters_{ticker}.pkl"
    loaded_model_parameters = load_prophet_model_parameters(file_path)
    
    # Prophet 모델 생성 및 파라미터 설정
    loaded_model = Prophet(
        changepoint_prior_scale=loaded_model_parameters['changepoint_prior_scale'],
        seasonality_prior_scale=loaded_model_parameters['seasonality_prior_scale'],
        holidays_prior_scale=loaded_model_parameters['holidays_prior_scale'],
        seasonality_mode=loaded_model_parameters['seasonality_mode'],
        changepoint_range=loaded_model_parameters['changepoint_range'],
        yearly_seasonality=loaded_model_parameters['yearly_seasonality'],
        weekly_seasonality=loaded_model_parameters['weekly_seasonality'],
        daily_seasonality=loaded_model_parameters['daily_seasonality'],
        growth=loaded_model_parameters['growth'],
        n_changepoints=loaded_model_parameters['n_changepoints']
    )
    loaded_model.add_country_holidays(country_name='US')
    
    loaded_model.add_regressor('predict')
    
    # 데이터 학습
    loaded_model.fit(finance_df)
    future = loaded_model.make_future_dataframe( periods = 1)
    future['predict'] = result_filtered_future['predict'].values
    # 주가 예측
    forecast = loaded_model.predict(future)

    return finance_df, forecast 