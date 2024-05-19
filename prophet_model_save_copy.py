#!/usr/bin/env python
# coding: utf-8

# In[5]:


from numba import cuda

device = cuda.get_current_device(); device.reset()


# In[6]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import os
import urllib.request
from tqdm import tqdm
from transformers import BertTokenizer, TFBertForSequenceClassification


# In[7]:


urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv", filename="finance_data.csv")


# In[8]:


data = pd.read_csv('finance_data.csv')
print('총 샘플의 수 :',len(data))


# In[9]:


data


# In[10]:


data['labels'] = data['labels'].replace(['negative', 'neutral', 'positive'],[0, 1, 2])
data[:5]


# In[11]:


del data['kor_sentence']


# In[12]:


data[:5]


# In[13]:


data.info()


# In[14]:


print('결측값 여부 :',data.isnull().values.any())


# In[15]:


print('kor_sentence 열의 유니크한 값 :',data['sentence'].nunique())


# In[16]:


duplicate = data[data.duplicated()]


# In[17]:


duplicate


# In[18]:


# 중복 제거
data.drop_duplicates(subset=['sentence'], inplace=True)
print('총 샘플의 수 :',len(data))


# In[19]:


data['labels'].value_counts().plot(kind='bar')


# In[20]:


print('레이블의 분포')
print(data.groupby('labels').size().reset_index(name='count'))


# In[21]:


print(f'중립의 비율 = {round(data["labels"].value_counts()[1]/len(data) * 100,3)}%')
print(f'긍정의 비율 = {round(data["labels"].value_counts()[2]/len(data) * 100,3)}%')
print(f'부정의 비율 = {round(data["labels"].value_counts()[0]/len(data) * 100,3)}%')


# In[22]:


data


# In[23]:


X_data = data['sentence']
y_data = data['labels']
print('본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)


# In[25]:


print('훈련 샘플의 개수 :', len(X_train))
print('테스트 샘플의 개수 :', len(X_test))


# In[26]:


print('--------훈련 데이터의 비율-----------')
print(f'중립 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%')
print(f'긍정 = {round(y_train.value_counts()[2]/len(y_train) * 100,3)}%')
print(f'부정 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%')


# In[27]:


print('--------테스트 데이터의 비율-----------')
print(f'중립 = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%')
print(f'긍정 = {round(y_test.value_counts()[2]/len(y_test) * 100,3)}%')
print(f'부정 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%')


# In[28]:


max_seq_len = 128


# In[29]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[30]:


tokenizer


# In[31]:


def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        # input_id는 워드 임베딩을 위한 문장의 정수 인코딩
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)

        # attention_mask는 실제 단어가 위치하면 1, 패딩의 위치에는 0인 시퀀스.
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count

        # token_type_id은 세그먼트 인코딩 -> 문장 2개를 분류나 해석하는 task에선 각 문장이 0과 1로 분류됨.
        token_type_id = [0] * max_seq_len

        # 길이가 다를시 오류메세지 띄워줌.
        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels


# In[32]:


X_train, y_train, max_seq_len


# In[33]:


train_X, train_y = convert_examples_to_features(X_train, y_train, max_seq_len=max_seq_len, tokenizer=tokenizer)


# In[34]:


train_X, train_y


# In[35]:


test_X, test_y = convert_examples_to_features(X_test, y_test, max_seq_len=max_seq_len, tokenizer=tokenizer)


# In[36]:


input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))
print('레이블 :',label)


# In[37]:


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW


# In[38]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# In[39]:


len(train_X[0]), len(train_y)


# In[40]:


# create Tensor datasets
train_data = TensorDataset(torch.tensor(train_X[0]), torch.tensor(train_X[1]), torch.tensor(train_X[2]), torch.tensor(train_y))
valid_data = TensorDataset(torch.tensor(test_X[0]), torch.tensor(test_X[1]), torch.tensor(test_X[2]), torch.tensor(test_y))

# dataloaders
batch_size = 32

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


# In[41]:


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


# In[42]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


# In[43]:


# epochs = 5
# valid_loss_min = np.Inf

# train_loss = torch.zeros(epochs)
# valid_loss = torch.zeros(epochs)

# train_acc = torch.zeros(epochs)
# valid_acc = torch.zeros(epochs)

# model.to(device)
# for e in tqdm(range(0, epochs)):
#     model.train()
#     # initialize hidden state 
#     # h = model.init_hidden(batch_size)
#     for inputs in tqdm(train_loader):
#         input_ids = inputs[0].to(device)
#         attention_mask = inputs[1].to(device)
#         labels = inputs[3].type(torch.LongTensor).to(device)
#         optimizer.zero_grad()
#         output = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = output.loss
#         train_loss[e] += loss.item()
#         loss.backward()
        
#         # calculating accuracy
#         # accuracy = acc(output,labels)
#         ps = F.softmax(output.logits, dim=1)
#         top_p, top_class = ps.topk(1, dim=1)
#         equals = top_class == labels.reshape(top_class.shape)
#         train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
        
#         #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         # nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#     train_loss[e] /= len(train_loader)
#     train_acc[e] /= len(train_loader)

    
#     model.eval()
#     for inputs in tqdm(valid_loader):
#         input_ids = inputs[0].to(device)
#         attention_mask = inputs[1].to(device)
#         labels = inputs[3].type(torch.LongTensor).to(device)

#         output = model(input_ids, attention_mask=attention_mask, labels=labels)
#         val_loss = output.loss
#         valid_loss[e] += val_loss.item()

#         ps = F.softmax(output.logits, dim=1)
#         top_p, top_class = ps.topk(1, dim=1)
#         equals = top_class == labels.reshape(top_class.shape)
#         valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
#     valid_loss[e] /= len(valid_loader)
#     valid_acc[e] /= len(valid_loader)
    
#     print(f'Epoch {e+1}') 
#     print(f'train_loss : {train_loss[e]}, val_loss : {valid_loss[e]}')
#     print(f'train_accuracy : {train_acc[e]*100}, val_accuracy : {valid_acc[e]*100}')
#     if valid_loss[e] <= valid_loss_min:
#         torch.save(model, 'bert_model.pth')
#         torch.save(model.state_dict(), 'bert_model_state_dict.pt')
#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss[e]))
#         valid_loss_min = valid_loss[e]
#     print(25*'==')


# In[44]:


import os
import pandas as pd

path = "C:/Users/chan/Documents/GitHub/ML/test10_AVGO/"
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

news_df


# In[45]:


# import os
# import pandas as pd

# path = "C:/Users/chan/Documents/GitHub/ML/articles/"
# news_df = pd.DataFrame(columns=["title", "content"])


# for txts in os.listdir(path):
    
#     full_path = os.path.join(path, txts)  # 파일 전체 경로 생성
#     if os.path.isfile(full_path):  # 파일인지 확인
#         with open(full_path, "r", encoding="utf-8") as txt_file:
#             title = txt_file.readline().strip()
#             content = txt_file.read().replace('\n', ' ')
#             news_df = news_df.append({"title": title, "content": content}, ignore_index=True)
# news_df


# In[46]:


# 데이터프레임의 'title'과 'content'를 이용하여 입력 데이터 생성
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


# In[47]:


predict_model = torch.load("bert_model_loss0.34.pth", map_location=torch.device('cpu'))
predict_model.to(device)


# In[48]:


predicted_labels = []

for inputs in tqdm(zip(input_ids, attention_masks)):
    input_ids = inputs[0].to(device)
    attention_mask = inputs[1].to(device)
    output = predict_model(input_ids, attention_mask=attention_mask)

    ps = F.softmax(output.logits, dim=1)
    top_p, top_class = ps.topk(1, dim=1)
    predicted_labels.append(top_class.item())

predict_df = pd.DataFrame({'predicted_label': predicted_labels})
news_df["predict"] = predict_df
news_df.to_csv('predicted_news.csv', index=False)


# In[49]:


news_df


# In[50]:


import os
import pandas as pd

# 데이터프레임과 데이터 로드 부분은 생략합니다.

# 'title' 열에서 날짜와 시간을 분리
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


# In[51]:


result_df = grouped_df[['date', 'predict']]
result_df


# In[52]:


import os
import pandas as pd

# 데이터프레임과 데이터 로드 부분은 생략합니다.

# 'title' 열에서 날짜와 시간을 분리
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
result_df = complete_df[['date', 'predict']]
result_df


# <h1>기간범위중 맨앞 데이터가 없을경우 1로채운다
# 

# In[56]:


from datetime import datetime, timedelta

start_date = datetime.strptime('2021-04-06', '%Y-%m-%d')
end_date = datetime.strptime('2024-05-10', '%Y-%m-%d')

# 주어진 범위 내에 있는 날짜 생성
missing_dates = pd.date_range(start=start_date, end=end_date).difference(result_df['date'])

# 새로운 날짜와 예측값 추가
for date in missing_dates:
    result_df = result_df.append({'date': date, 'predict': 1.0}, ignore_index=True)

# 날짜 기준으로 데이터프레임 정렬
result_df = result_df.sort_values('date').reset_index(drop=True)
result_df


# In[57]:


news_df['predict'].value_counts().plot(kind='bar')


# In[59]:


import numpy as np
import pandas as pd

import pandas_datareader.data as web # 주식 데이터를 얻어오기 위해 사용
import datetime # 시간 처리
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from prophet import Prophet
get_ipython().run_line_magic('matplotlib', 'inline')

# 데이터를 가져오고 나서, 이동평균을 구해야함.
# 국내 종목 : 삼성전자

# 날짜 : 3년 간 삼성전자 주가 분석(2017.01.02) ~ (2021.06.07)
start = datetime.datetime( 2021, 4, 6 )
end = datetime.datetime( 2024, 5, 10)


# In[60]:


def makeStockChart(code, sDay, eDay):
    """
    이 함수는 종목코드와 조회 시작일, 종료일을 넣으면 차트를 그려준다.
    이동 평균선은 5일, 20일, 60일, 120일을 지원한다.
    """
    
    # 데이터 가져오기
    data = yf.download(code, start=sDay, end=eDay)
    
    # 이동 평균 계산
    data['5MA'] = data['Adj Close'].rolling(window=5).mean()
    data['20MA'] = data['Adj Close'].rolling(window=20).mean()
    data['60MA'] = data['Adj Close'].rolling(window=60).mean()
    data['120MA'] = data['Adj Close'].rolling(window=120).mean()
    
    # 차트 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close')
    plt.plot(data['5MA'], label='5MA')
    plt.plot(data['20MA'], label='20MA')
    plt.plot(data['60MA'], label='60MA')
    plt.plot(data['120MA'], label='120MA')
    plt.title(f"{code}'s Stock Chart")
    plt.legend()
    plt.show()


# In[61]:


code_name = 'AVGO'
makeStockChart(code_name, start, end)


# In[62]:


datas = yf.download(code_name, start=start, end=end)

# 컬럼 ds(YYYY-MM-DD), y(종가 : Close) 구성
# 해당 구저의 DateFrame만들기 위한 딕셔너리 선언

dic = {
    'ds' : datas.index,
    'y' : datas.Close,
    # 'volume': datas.Volume
    
}

# ds 와 y를 컬럼으로 갖는 데이터 프레임 생성
finance_df = pd.DataFrame( dic )

#인덱스 초기화(원본 까지 적용)
finance_df.reset_index( inplace=True )

# 'ds' 컬럼과 중복되는 'Date' 컬럼 제거
del finance_df['Date']

# 페이스북의 시계열 예측 모델에 사용한 데이터 준비
finance_df.head(5)
finance_df.tail(5)
# print(len(finance_df))
finance_df


# In[63]:


import pandas as pd
import datetime

# 정해진 기간을 데이터프레임화


# start와 end 사이의 날짜를 포함하는 DataFrame 생성
date_range = pd.date_range(start=start, end=end)
df_date_range = pd.DataFrame(date_range, columns=['date'])

# 다른 데이터프레임 예시 생성


# df_date_range에서 other_dates의 date와 겹치는 날짜를 제거
df_filtered = df_date_range[~df_date_range['date'].isin(finance_df['ds'])]
df_filtered #미국주식 안열리는 날 계산한 데이터 프레임


# In[64]:


# 겹치는 날짜 
common_dates = result_df[result_df['date'].isin(df_filtered['date'])]['date']

# 겹치는 날짜를 갖는 행 제거
result_filtered = result_df[~result_df['date'].isin(common_dates)]
# result_filtered = result_filtered.iloc[:-1]
result_filtered


# In[65]:


result_filtered.reset_index(inplace=True)
result_filtered


# In[66]:


result_filtered_future = result_filtered.copy()
result_filtered_future = result_filtered_future.drop(columns=['index'])
result_filtered_future


# In[67]:


# import pandas as pd

# # finance_df와 result_filtered_future 데이터프레임을 병합하여 새로운 데이터프레임 생성
# result_filtered_future = pd.concat([result_filtered_future, finance_df['volume']], axis=1)

# # 결과 확인
# result_filtered_future


# <h1>future -> forecast를 위한 값 추가</h1>

# In[68]:


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

result_filtered_future


# In[69]:


finance_df = pd.concat([finance_df, result_filtered['predict']], axis=1)

finance_df


# In[62]:


# finance_df


# In[70]:


import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Day 단위로 데이터가 구성되어 있으므로, 일 단위 주기성 활성화
# model = Prophet(changepoint_prior_scale=0.31150996767072153,
#                 changepoint_range=0.8409742702125196, 
#                 daily_seasonality=True,
#                 holidays_prior_scale=0.5429118795932729, 
#                 seasonality_mode='multiplicative',
#                 seasonality_prior_scale=1.2964727674363865,
#                 weekly_seasonality=True,
#                 n_changepoints=25,
#                 growth='linear',
#                 yearly_seasonality=True)
# model = Prophet(changepoint_prior_scale=0.3225469441035442,
#                 changepoint_range=0.8959729377163332, 
#                 daily_seasonality=True,
#                 holidays_prior_scale=0.48141493972833566, 
#                 seasonality_mode='multiplicative',
#                 seasonality_prior_scale=0.27212358941114123,
#                 weekly_seasonality=False,
#                 n_changepoints=24,
#                 growth='linear',
#                 yearly_seasonality=True) # cross_validation(m, initial='250 days', period='10 days', horizon='1 days')
model = Prophet(changepoint_prior_scale=0.1541135741310533,
                changepoint_range=0.7075492829641814, 
                daily_seasonality=False,
                holidays_prior_scale=0.5080563204252423, 
                seasonality_mode='multiplicative',
                seasonality_prior_scale=4.043977713268082,
                weekly_seasonality=True,
                n_changepoints=25,
                growth='linear',
                yearly_seasonality=True) #cross_validation(m, initial='250 days', period='20 days', horizon='1 days')
                
model.add_country_holidays(country_name='US')
model.add_regressor('predict')
# 데이터 학습 시작 -> 기계학습
model.fit( finance_df )
# print(model.params['beta'])
# 주가 예측 위한 날짜 데이터 세팅 -> 기존 데이터 + 향후 10일치 예측값

future = model.make_future_dataframe( periods = 1)
future['predict'] = result_filtered_future['predict'].values
# 주가 예측
forecast = model.predict( future )

# forecast.columns ->
'''
  Index(['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'daily', 'daily_lower', 'daily_upper', 'weekly', 'weekly_lower',
       'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
       'multiplicative_terms', 'multiplicative_terms_lower',
       'multiplicative_terms_upper', 'yhat'],
      dtype='object')
''' 

# 모델 예측 결과 출력
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(11))

# 모델 예측 그래프 출력
fig = model.plot(forecast)

plt.show(fig)


# <h1>prophet 하이퍼파라미터 설정 model로 예측 돌리기</h1>

# In[64]:


# import yfinance as yf
# import datetime

# # Tesla 주식에 대한 데이터를 가져오기
# tsla = yf.Ticker("TSLA")
# tsla_income = tsla.income_stmt.T
# tsla_balance = tsla.balance_sheet.T
# tsla_cash = tsla.cashflow.T
# tsla_combined = pd.concat([tsla_income, tsla_balance, tsla_cash], axis=1)

# # 필요한 열만 선택하여 새로운 데이터프레임 생성
# tsla_selected = tsla_combined[["Net Income Common Stockholders", "Basic EPS", "Working Capital", "Operating Cash Flow"]]

# # 결과 출력
# tsla_selected = tsla_selected.fillna(0)
# tsla_selected




# In[65]:


# from prophet.diagnostics import cross_validation, performance_metrics
# import matplotlib.pyplot as plt
# model = Prophet( changepoint_prior_scale= 0.4469497270584068, 
# changepoint_range= 0.8117567477475541, 
# daily_seasonality= True,
# seasonality_mode='multiplicative',
# growth= 'linear', 
# holidays_prior_scale= 0.05160382047585584, 
# n_changepoints= 22, 
# seasonality_prior_scale= 0.3523031728392127, 
# weekly_seasonality= False, 
# yearly_seasonality= True
#                ) 

# model.add_country_holidays(country_name='US')
# model.add_regressor('predict')
# # 데이터와 Prophet 모델 설정 부분은 생략하고, 모델 학습 부분부터 시작
# model.fit(finance_df)

# # 크로스 밸리데이션 설정
# df_cv = cross_validation(model, initial='20 days', period='2 days', horizon='1 days')

# # 크로스 밸리데이션 결과의 성능 지표 계산
# df_p = performance_metrics(df_cv)
# print(df_p[['rmse', 'mae','mape']])


# In[ ]:


# import pickle

# # 모델 객체를 파일로 저장
# with open("prophet_model.pkl", "wb") as f:
#     pickle.dump(model, f)


# In[ ]:


# # 저장된 모델 파일을 로드
# with open("prophet_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)


# <h1>buy,sell 출력 json파일 만들 코드

# In[ ]:


# initial_capital = 1000.0  # 초기 자본
# current_capital = initial_capital  # 현재 자본, 거래 과정에서 업데이트됨
# shares_owned = 0
# position = 'None'  # 현재 포지션 상태
# day = 1
# month_day = 30
# # 시뮬레이션 수익 계산
# profit = 0.0

# actual_prices = finance_df.set_index('ds')['y'][-(day+31):-(day)]  # 마지막 31일 실제 가격
# predicted_prices = forecast.set_index('ds')['yhat'][-(day+31):-(day)]

# dates = [actual_prices.index[0] - pd.Timedelta(days=1)]  # 첫 거래일 이전 날짜 추가
# capitals = [initial_capital]

# # 날짜와 함께 매도, 매수, 보유 결과를 출력하기 위해 수정
# for i in range(month_day):
#     today_date = actual_prices.index[i]
#     today_price = actual_prices.iloc[i]
#     predicted_next_day_price = predicted_prices.iloc[i]

#     # 조건을 검사하고 매수, 매도, 보유를 결정
#     if i < (month_day-1):  # 마지막 날짜를 제외한 모든 날에 대해 검사
#         next_day_price = actual_prices.iloc[i+1]
#         predicted_change = predicted_next_day_price - today_price

#         # 매수 조건
#         if predicted_change > 0 and position != 'Hold':
#             shares_owned = current_capital / today_price
#             current_capital = 0
#             position = 'Hold'
#             print(f"{today_date} - 매수: {today_price} (실제 다음날 가격: {next_day_price:,.3f}, 예측된 다음날 가격: {predicted_next_day_price:,.3f})")

#         # 매도 조건
#         elif predicted_change < 0 and position == 'Hold':
#             current_capital = shares_owned * next_day_price
#             shares_owned = 0
#             position = 'None'
#             print(f"{today_date} - 매도: {next_day_price} (실제 다음날 가격: {next_day_price:,.3f}, 예측된 다음날 가격: {predicted_next_day_price:,.3f})")

#         # 보유 조건
#         elif position == 'Hold' and abs(predicted_change) <= 0.01 * today_price:
#             print(f"{today_date} - 보유: {today_price}")
#         else:
#     # 보유 또는 기타 조건 로직
#             if position == 'Hold':
#                 print(f"{today_date} - 보유: 현재 가격 {today_price:,.3f}, 예측된 다음날 가격 {predicted_next_day_price:,.3f}")
#             else:
#                 print(f"{today_date} - 조치 없음: 현재 가격 {today_price:,.3f}, 예측된 다음날 가격 {predicted_next_day_price:,.3f}")
#     dates.append(today_date)
#     capitals.append(current_capital + (shares_owned * today_price))
# # 마지막 날 주식을 모두 매도하고 최종 자본 계산
# if position == 'Hold' and shares_owned > 0:
#     final_price = actual_prices.iloc[-1]
#     current_capital = shares_owned * final_price
#     shares_owned = 0
#     print(f"{actual_prices.index[-1]} - 마지막 날 매도: {final_price:,.3f}")

# profit = current_capital - initial_capital  # 최종 이익 계산
# return_rate = ((current_capital - initial_capital) / initial_capital) * 100
# # 결과 출력
# print(f"초기 자본: ${initial_capital:,.3f}")
# print(f"최종 자본: ${current_capital:,.3f}")
# print(f"수익: ${profit:,.3f}")
# print(f"수익률: {return_rate:.2f}%")


# In[ ]:


# import pandas as pd

# # 초기 자본 설정
# initial_capital = 1000.0

# # 사용자로부터 시작일과 종료일 입력 받기
# start_date = '2023-03-05'
# end_date_1 = '2024-04-04'
# end_date_2 = '2024-04-05'

# # 입력된 날짜 범위로 데이터 필터링
# filtered_df = finance_df[(finance_df['ds'] >= start_date) & (finance_df['ds'] <= end_date_1)]
# filtered_df_forcast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date_2)]
# # 현재 자본 및 보유 주식 수 초기화
# current_capital = initial_capital
# shares_owned = 0

# # 현재 포지션 초기화
# position = 'None'

# # 실제 및 예측 주식 가격 설정
# actual_prices = filtered_df.set_index('ds')['y']
# predicted_prices = filtered_df_forcast.set_index('ds')['yhat']

# # 시뮬레이션 시작
# for date, today_price, predicted_next_day_price in zip(actual_prices.index, actual_prices.values, predicted_prices.shift(-1).values):
#     predicted_change = predicted_next_day_price - today_price

#     # 매수 조건
#     if predicted_change > 0 and position != 'Hold':
#         shares_to_buy = current_capital / today_price
#         shares_owned += shares_to_buy
#         current_capital -= shares_to_buy * today_price
#         position = 'Hold'
#         print(f"{date} - 매수: ${today_price:.3f} (예측 다음날 가격: ${predicted_next_day_price:.3f})")

#     # 매도 조건
#     elif predicted_change < 0 and position == 'Hold':
#         current_capital += shares_owned * today_price
#         shares_owned = 0
#         position = 'None'
#         print(f"{date} - 매도: ${today_price:.3f} (예측 다음날 가격: ${predicted_next_day_price:.3f})")

#     # 보유 조건
#     elif position == 'Hold' and abs(predicted_change) <= 0.01 * today_price:
#         print(f"{date} - 보유: ${today_price:.3f}")
#     else:
#         # 조치 없음 또는 보유
#         if position == 'Hold':
#             print(f"{date} - 보유: 현재 가격 ${today_price:.3f}, 예측 다음날 가격 ${predicted_next_day_price:.3f}")
#         else:
#             print(f"{date} - 조치 없음: 현재 가격 ${today_price:.3f}, 예측 다음날 가격 ${predicted_next_day_price:.3f}")

# # 최종 자본 및 수익률 계산
# final_capital = current_capital + (shares_owned * actual_prices.iloc[-1])
# profit = final_capital - initial_capital
# return_rate = (profit / initial_capital) * 100

# # 결과 출력
# print(f"초기 자본: ${initial_capital:.3f}")
# print(f"최종 자본: ${final_capital:.3f}")
# print(f"수익: ${profit:.3f}")
# print(f"수익률: {return_rate:.3f}%")


# In[ ]:


# print(actual_prices)
# predicted_prices.shift(-1).values
# predicted_prices


# In[84]:


from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np




param_space = {
    'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),  # changepoint_prior_scale 범위를 수정
    'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10)),   # seasonality_prior_scale 범위를 수정
    'holidays_prior_scale': hp.loguniform('holidays_prior_scale', np.log(0.01), np.log(10)),         # holidays_prior_scale 범위를 수정
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
    'changepoint_range': hp.uniform('changepoint_range', 0.7, 0.95),
    'yearly_seasonality': hp.choice('yearly_seasonality', [True, False]),
    'weekly_seasonality': hp.choice('weekly_seasonality', [True, False]),
    'daily_seasonality': hp.choice('daily_seasonality', [True, False]),
    'n_changepoints': hp.quniform('n_changepoints', 10, 25, 1)
}

# 최적화할 목적 함수 정의
def objective(params):
    m = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        changepoint_range=params['changepoint_range'],
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        daily_seasonality=params['daily_seasonality'],
        growth='linear',
        n_changepoints=int(params['n_changepoints'])
    )
    m.add_country_holidays(country_name='US')
    m.fit(finance_df)
    
    # 크로스 밸리데이션을 통해 성능 평가
    df_cv = cross_validation(m, initial='250 days', period='20 days', horizon='1 days')
    df_p = performance_metrics(df_cv)
    # MAPE를 최소화
    mape = df_p['mape'].mean()
    
    
    # hyperopt는 최소화 문제를 해결하므로, MAPE를 반환
    return {'loss': mape, 'status': STATUS_OK}

# 최적화 실행
trials = Trials()
best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best params:", best_params)


# In[85]:


# import logging
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
# import numpy as np

# # cmdstanpy 로그 수준 설정
# logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# # Prophet 모델 생성 및 하이퍼파라미터 설정
# # 'seasonality_mode' 값을 숫자에서 문자열로 매핑
# seasonality_modes = ['additive', 'multiplicative']
# seasonality_mode = seasonality_modes[best_params['seasonality_mode']]

# yearly_seasonalitys = [True,False]
# yearly_seasonality = yearly_seasonalitys[best_params['yearly_seasonality']]
# weekly_seasonalitys = [True,False]
# weekly_seasonality = weekly_seasonalitys[best_params['weekly_seasonality']]
# daily_seasonalitys = [True,False]
# daily_seasonality = daily_seasonalitys[best_params['daily_seasonality']]

# # Prophet 모델 생성 및 하이퍼파라미터 설정
# model = Prophet(
#     changepoint_prior_scale = best_params['changepoint_prior_scale'],
#     seasonality_prior_scale = best_params['seasonality_prior_scale'],
#     holidays_prior_scale = best_params['holidays_prior_scale'],
#     seasonality_mode = seasonality_mode,
#     changepoint_range = best_params['changepoint_range'],
#     yearly_seasonality = yearly_seasonality,
#     weekly_seasonality = weekly_seasonality,
#     daily_seasonality = daily_seasonality,
#     growth='linear',
#     n_changepoints = int(best_params['n_changepoints'])
# )
# model.add_country_holidays(country_name='US')
# model.add_regressor('predict')
# # 데이터 학습 시작 -> 기계학습

# model.fit( finance_df )
# # print(model.params['beta'])
# # 주가 예측 위한 날짜 데이터 세팅 -> 기존 데이터 + 향후 10일치 예측값

# future = model.make_future_dataframe( periods = 1)
# future['predict'] = result_filtered_future['predict'].values
# # 주가 예측
# forecast = model.predict(future)

# # forecast.columns ->
# '''
#   Index(['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
#        'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
#        'daily', 'daily_lower', 'daily_upper', 'weekly', 'weekly_lower',
#        'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
#        'multiplicative_terms', 'multiplicative_terms_lower',
#        'multiplicative_terms_upper', 'yhat'],
#       dtype='object')
# ''' 

# # 모델 예측 결과 출력
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(11))

# # 모델 예측 그래프 출력
# fig = model.plot(forecast)

# plt.show(fig)


# In[93]:


import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

# cmdstanpy 로그 수준 설정
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Prophet 모델 생성 및 하이퍼파라미터 설정
# 'seasonality_mode' 값을 숫자에서 문자열로 매핑
seasonality_modes = ['additive', 'multiplicative']
seasonality_mode = seasonality_modes[best_params['seasonality_mode']]

yearly_seasonalitys = [True,False]
yearly_seasonality = yearly_seasonalitys[best_params['yearly_seasonality']]
weekly_seasonalitys = [True,False]
weekly_seasonality = weekly_seasonalitys[best_params['weekly_seasonality']]
daily_seasonalitys = [True,False]
daily_seasonality = daily_seasonalitys[best_params['daily_seasonality']]

# Prophet 모델 생성 및 하이퍼파라미터 설정
model_parameter = {
    'changepoint_prior_scale' : best_params['changepoint_prior_scale'],
    'seasonality_prior_scale' : best_params['seasonality_prior_scale'],
    'holidays_prior_scale' : best_params['holidays_prior_scale'],
    'seasonality_mode' : seasonality_mode,
    'changepoint_range' : best_params['changepoint_range'],
    'yearly_seasonality' : yearly_seasonality,
    'weekly_seasonality' : weekly_seasonality,
    'daily_seasonality' : daily_seasonality,
    'growth':'linear',
    'n_changepoints' : int(best_params['n_changepoints'])
}

with open('prophet_model_parameters.pkl', 'wb') as f:
    pickle.dump(model_parameter, f)


# In[97]:


# loaded_model_parameters['changepoint_prior_scale']


# In[95]:


# 저장된 모델 파라미터를 파일로 저장


# 저장된 모델 파라미터를 불러오는 함수
def load_prophet_model_parameters(file_path):
    with open(file_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params
    
loaded_model_parameters = load_prophet_model_parameters('prophet_model_parameters.pkl')

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

# forecast.columns ->
'''
  Index(['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'daily', 'daily_lower', 'daily_upper', 'weekly', 'weekly_lower',
       'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
       'multiplicative_terms', 'multiplicative_terms_lower',
       'multiplicative_terms_upper', 'yhat'],
      dtype='object')
''' 

# 모델 예측 결과 출력
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(11))

# 모델 예측 그래프 출력
fig = loaded_model.plot(forecast)

plt.show(fig)


# In[96]:


finance_df


# In[68]:


# import pickle

# # 모델 객체를 파일로 저장
# with open("prophet_model_AVGO.pkl", "wb") as f:
#     pickle.dump(model, f)


# In[98]:


# with open("prophet_model_AVGO.pkl", "rb") as f:
#     model = pickle.load(f)

# # 크로스 밸리데이션 설정
# df_cv = cross_validation(model, initial='250 days', period='20 days', horizon='1 days')

# # 크로스 밸리데이션 결과의 성능 지표 계산
# df_p = performance_metrics(df_cv)
# print(df_p[['rmse', 'mae','mape']])


# In[99]:


# import pickle
# import logging
# import matplotlib.pyplot as plt
# from prophet import Prophet

# # cmdstanpy 로그 수준 설정
# logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# # 저장된 모델 파일을 로드
# with open("prophet_model_AVGO.pkl", "rb") as f:
#     model = pickle.load(f)

# # 주가 예측 위한 날짜 데이터 세팅 -> 기존 데이터 + 향후 10일치 예측값
# future = model.make_future_dataframe(periods=1)
# future['predict'] = result_filtered_future['predict'].values[len(result_filtered_future['predict'].values)-756:]

# # 주가 예측
# forecast = model.predict(future)

# # 모델 예측 결과 출력
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(11))

# # 모델 예측 그래프 출력
# fig = model.plot(forecast)
# plt.show(fig)

# print(finance_df.tail(11))
# print(future['predict'].tail(11))


# In[ ]:


# from prophet.diagnostics import cross_validation, performance_metrics
# import matplotlib.pyplot as plt
# model = Prophet( changepoint_prior_scale= 0.4469497270584068, 
# changepoint_range= 0.8117567477475541, 
# daily_seasonality= True,
# seasonality_mode='multiplicative',
# growth= 'linear', 
# holidays_prior_scale= 0.05160382047585584, 
# n_changepoints= 22, 
# seasonality_prior_scale= 0.3523031728392127, 
# weekly_seasonality= False, 
# yearly_seasonality= True
#                ) 

# model.add_country_holidays(country_name='US')
# model.add_regressor('predict')
# # 데이터와 Prophet 모델 설정 부분은 생략하고, 모델 학습 부분부터 시작
# model.fit(finance_df)

# # 크로스 밸리데이션 설정
# df_cv = cross_validation(model, initial='20 days', period='2 days', horizon='1 days')

# # 크로스 밸리데이션 결과의 성능 지표 계산
# df_p = performance_metrics(df_cv)
# print(df_p[['rmse', 'mae','mape']])


# In[100]:


import pandas as pd

# 초기 자본 설정
initial_capital = 100000000.0


# def trading():
# 사용자로부터 시작일과 종료일 입력 받기
start_date = '2024-04-05'
end_date_1 = '2024-05-09'
end_date_2 = '2024-05-10'

# 입력된 날짜 범위로 데이터 필터링
filtered_df = finance_df[(finance_df['ds'] >= start_date) & (finance_df['ds'] <= end_date_1)]
filtered_df_forcast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date_2)]
# 현재 자본 및 보유 주식 수 초기화
current_capital = initial_capital
shares_owned = 0

# 현재 포지션 초기화
position = 'None'

# 실제 및 예측 주식 가격 설정
actual_prices = filtered_df.set_index('ds')['y']
predicted_prices = filtered_df_forcast.set_index('ds')['yhat']

# 시뮬레이션 시작
for date, today_price, predicted_next_day_price in zip(actual_prices.index, actual_prices.values, predicted_prices.shift(-1).values):
    predicted_change = predicted_next_day_price - today_price

    # 매수 조건
    if predicted_change > 0 and position != 'Hold':
        shares_to_buy = current_capital / today_price
        shares_owned += shares_to_buy
        current_capital -= shares_to_buy * today_price
        position = 'Hold'
        print(f"{date} - 매수: ${today_price:.3f} (예측 다음날 가격: ${predicted_next_day_price:.3f})")

    # 매도 조건
    elif predicted_change < 0 and position == 'Hold':
        current_capital += shares_owned * today_price
        shares_owned = 0
        position = 'None'
        print(f"{date} - 매도: ${today_price:.3f} (예측 다음날 가격: ${predicted_next_day_price:.3f})")

    # 보유 조건
    elif position == 'Hold' and abs(predicted_change) <= 0.01 * today_price:
        print(f"{date} - 보유: ${today_price:.3f}")
    else:
        # 조치 없음 또는 보유
        if position == 'Hold':
            print(f"{date} - 보유: 현재 가격 ${today_price:.3f}, 예측 다음날 가격 ${predicted_next_day_price:.3f}")
        else:
            print(f"{date} - 조치 없음: 현재 가격 ${today_price:.3f}, 예측 다음날 가격 ${predicted_next_day_price:.3f}")

# 최종 자본 및 수익률 계산
final_capital = current_capital + (shares_owned * actual_prices.iloc[-1])
profit = final_capital - initial_capital
return_rate = (profit / initial_capital) * 100

# 결과 출력
print(f"초기 자본: ${initial_capital:.3f}")
print(f"최종 자본: ${final_capital:.3f}")
print(f"수익: ${profit:.3f}")
print(f"수익률: {return_rate:.3f}%")


# In[101]:


def simulate_trading(start_date, end_date_1,end_date_2):
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
            trading_history.append((date, '매수', today_price, predicted_next_day_price))

        # 매도 조건
        elif predicted_change < 0 and position == 'Hold':
            current_capital += shares_owned * today_price
            shares_owned = 0
            position = 'None'
            trading_history.append((date, '매도', today_price, predicted_next_day_price))

        # 보유 조건
        elif position == 'Hold' and abs(predicted_change) <= 0.01 * today_price:
            trading_history.append((date, '보유', today_price, predicted_next_day_price))

        # 조치 없음 또는 보유
        else:
            if position == 'Hold':
                trading_history.append((date, '보유', today_price, predicted_next_day_price))
            else:
                trading_history.append((date, '조치 없음', today_price, predicted_next_day_price))

    # 최종 자본 및 수익률 계산
    final_capital = current_capital + (shares_owned * actual_prices.iloc[-1])
    profit = final_capital - initial_capital
    return_rate = (profit / initial_capital) * 100

    return trading_history, final_capital, return_rate

# 주어진 날짜 범위로 거래 시뮬레이션 실행
start_date = '2024-04-05'
end_date_1 = '2024-05-09'
end_date_2 = '2024-05-10'
trading_history, final_capital, return_rate = simulate_trading(start_date, end_date_1,end_date_2)

# 거래 기록 출력
# for record in trading_history:
#     print(f"{record[0]} - {record[1]}: 현재 가격 ${record[2]:.3f}, 예측 다음날 가격 ${record[3]:.3f}")

# 최종 자본 및 수익률 출력
print(f"최종 자본: ${final_capital:.2f}")
print(f"수익률: {return_rate:.2f}%")


# In[104]:


def simulate_trading(start_date, end_date_1, end_date_2):
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
            trading_history.append('Cell')

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

# 함수 호출하여 마지막 인덱스 값만 출력
# start_date = '2024-04-05'
# end_date_1 = '2024-05-09'
# end_date_2 = '2024-05-10'
# last_trading_record = simulate_trading(start_date, end_date_1, end_date_2)
# print(last_trading_record)

# export default simulate_trading


# In[ ]:


# end_date_basic = input('날짜를 입력하게요 ex)2024-05-07')
# end_date_future = input('날짜를 입력하게요 ex)2024-05-07')
# trainding(end_date_basic,end_date_future)

