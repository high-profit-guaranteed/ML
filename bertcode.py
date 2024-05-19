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

import pandas_datareader.data as web # 주식 데이터를 얻어오기 위해 사용
import datetime # 시간 처리
import yfinance as yf
import FinanceDataReader as fdr
from prophet import Prophet
# %matplotlib inline

def bertcode_predict(ticker, start_date, end_date_1,counter):
    ticker = ticker
    counter = counter
    #start_date 21-04-06 / end_date_1 24-05-09
    start_date = start_date
    end_date = end_date_1

    start_year, start_month, start_day = start_date.split('-')
    end_year, end_month, end_day = end_date.split('-')
    
    start_year = int(start_year)
    start_month = int(start_month)
    start_day = int(start_day)

    end_year = int(end_year)
    end_month = int(end_month)
    end_day = int(end_day)
    


    
    max_seq_len = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    path = f"C:/Users/chan/Documents/GitHub/ML/test_{ticker}/"
    # path = "C:/Users/chan/Documents/GitHub/ML/test10_AVGO/"
    news_df = pd.DataFrame(columns=["title", "content"])

    
    for txts in os.listdir(path):
        full_path = os.path.join(path, txts)  # 파일 전체 경로 생성
        if full_path.endswith('header.txt'):
            continue
        if os.path.isfile(full_path):  # 파일인지 확인
            with open(full_path, "r", encoding="utf-8") as txt_file:
                title = txt_file.readline().strip()
                content = txt_file.read().replace('\n', ' ')
                # DataFrame 생성 후 concat 함수를 사용하여 추가
                new_row = pd.DataFrame({"title": [title], "content": [content]})
                news_df = pd.concat([news_df, new_row], ignore_index=True)
    inputs = [content for content in news_df['content']]

    # 입력 데이터를 BERT 모델의 입력 형식에 맞게 변환
    max_length = 128
    input_ids = []
    attention_masks = []
    
    for content in inputs:
        encoded_dict = tokenizer.encode_plus(
                            content,                    # content
                            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks
                            return_tensors = 'pt',     # Return pytorch tensors
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    predict_model = torch.load("bert_model_loss0.34.pth", map_location=torch.device('cpu'))
    predict_model.to(torch.device('cpu'))

    predicted_labels = []

    for inputs in tqdm(zip(input_ids, attention_masks)):
        input_ids = inputs[0].to(torch.device('cpu'))
        attention_mask = inputs[1].to(torch.device('cpu'))
        output = predict_model(input_ids, attention_mask=attention_mask)
    
        ps = F.softmax(output.logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)
        predicted_labels.append(top_class.item())
    
    predict_df = pd.DataFrame({'predicted_label': predicted_labels})
    news_df["predict"] = predict_df
    news_df.to_csv('predicted_news.csv', index=False)

    news_df['date'] = pd.to_datetime(news_df['title']).dt.date  # 날짜 추출
    news_df['time'] = pd.to_datetime(news_df['title']).dt.time  # 시간 추출
    
    # 날짜별로 그룹화하고, 'predict' 열에 대한 평균을 계산
    grouped_df = news_df.groupby('date').agg({
        'title': lambda x: x.tolist(),
        'content': lambda x: x.tolist(),
        'predict': lambda x: round(x.mean(), 3)  
    })
    
    grouped_df.reset_index(inplace=True)  # 인덱스 리셋
    
    # 결과 출력
    grouped_df[['date', 'title', 'content', 'predict']]

    result_df = grouped_df[['date', 'predict']]

    news_df['date'] = pd.to_datetime(news_df['title']).dt.date  # 날짜 추출
    news_df['time'] = pd.to_datetime(news_df['title']).dt.time  # 시간 추출
    
    # 'date' 열을 datetime 형식으로 다시 변환
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    # 날짜별로 그룹화하고, 'predict' 열에 대한 평균을 계산
    grouped_df = news_df.groupby('date').agg({
        'title': lambda x: x.tolist(),
        'content': lambda x: x.tolist(),
        'predict': lambda x: round(x.mean(), 3)  
    })
    
    grouped_df.reset_index(inplace=True)  # 인덱스 리셋
    
    # 모든 날짜를 포함하는 날짜 범위 생성
    all_dates = pd.date_range(start=grouped_df['date'].min(), end=grouped_df['date'].max(), freq='D')
    # 새로운 DataFrame 생성 후 기존 데이터와 병합
    complete_df = pd.DataFrame(all_dates, columns=['date'])
    complete_df['date'] = pd.to_datetime(complete_df['date'])  # 날짜를 datetime으로 변환
    complete_df = complete_df.merge(grouped_df, on='date', how='left')
    # 누락된 'predict' 값을 이전 값으로 채우기
    complete_df['predict'] = complete_df['predict'].fillna(method='ffill')
    
    # 'date'와 'predict' 열만 선택하여 결과 출력
    result_df_1 = complete_df[['date', 'predict']]

    ##############################################
    csv_filename = f"news_data_{ticker}.csv"
    loaded_df = pd.read_csv(csv_filename)

    result_df_1['date'] = result_df_1['date'].dt.date
    result_df = pd.concat([loaded_df, result_df_1], ignore_index=True)
    # print(result_df)

    ###############################################

    start_date = datetime.datetime( start_year, start_month, start_day )
    end_date = datetime.datetime( end_year, end_month, end_day)
    
    # # 주어진 범위 내에 있는 날짜 생성
    # missing_dates = pd.date_range(start=start_date, end=end_date).difference(result_df['date'])
    
    # # 새로운 날짜와 예측값 추가
    # for date in missing_dates:
    #     result_df = result_df.append({'date': date, 'predict': 1.0}, ignore_index=True)
    
    # 날짜 기준으로 데이터프레임 정렬
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values('date').reset_index(drop=True)

    if counter == 6:
        result_df.to_csv(csv_filename, index=False)
        
    def makeStockChart(code, sDay, eDay):
        data = yf.download(code, start=sDay, end=eDay)

    code_name = ticker
    makeStockChart(code_name, start_date, end_date) #기간받아오기
    
    datas = yf.download(code_name, start=start_date, end=end_date)

    dic = {
        'ds' : datas.index,
        'y' : datas.Close,
        # 'volume': datas.Volume
        
    }
    
    finance_df = pd.DataFrame( dic )
    
    #인덱스 초기화(원본 까지 적용)
    finance_df.reset_index( inplace=True )
    
    del finance_df['Date']

    #########################
    stock = yf.Ticker(ticker)
    hist = stock.history(period = '1d',interval="1h")
    last_row = finance_df.iloc[-1]  # 마지막 행 가져오기
    next_date = last_row['ds'] + pd.DateOffset(days=1)
    second_price = hist['Close'][counter]# 1 -> 10시반 / 3 -> 12시반 / 5 -> 2시반 / 7 -> 4시반
    start_next_row = pd.DataFrame({'ds': [next_date], 'y': second_price })
    finance_df = pd.concat([finance_df, start_next_row], ignore_index=True)
    
    #########################
    
    date_range = pd.date_range(start=start_date, end=end_date)
    df_date_range = pd.DataFrame(date_range, columns=['date'])
    df_filtered = df_date_range[~df_date_range['date'].isin(finance_df['ds'])]

    common_dates = result_df[result_df['date'].isin(df_filtered['date'])]['date']
    result_filtered = result_df[~result_df['date'].isin(common_dates)]

    result_filtered.reset_index(inplace=True)

    result_filtered_future = result_filtered.copy()
    result_filtered_future = result_filtered_future.drop(columns=['index'])

    # 'predict' 열에만 평균 값 추가
    mean_last_seven = result_filtered_future['predict'].iloc[-7:].mean()
    # mean_last_seven_1 = result_filtered_future['volume'].iloc[-7:].mean()
    # 다음 날짜 계산
    last_date = result_filtered_future['date'].iloc[-1]
    next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
    
    # 데이터프레임에 새로운 행 추가
    new_row = {'date': next_date, 'predict': round(mean_last_seven, 3)}  # 소수점 세 자리까지 반올림
    result_filtered_future = result_filtered_future.append(new_row, ignore_index=True)
    
    # 'predict' 열의 소수점 자리 수 설정
    result_filtered_future['predict'] = result_filtered_future['predict'].round(3)
    
    finance_df = pd.concat([finance_df, result_filtered['predict']], axis=1)

    return finance_df, result_filtered_future
# print(bertcode_predict('AVGO', '2021-04-05','2024-05-16' ,1))


