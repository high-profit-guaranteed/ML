from fastapi import FastAPI
from trainding import simulate_trading
from googleNews import saveNews

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
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
from datetime import datetime, time
import concurrent.futures

app = FastAPI()



# @app.get('/avgo')
# async def avgo():

#     ticker = 'AVGO'
#     start_date = '2021-04-05'
#     end_date = '2024-05-15'
#     end_date_future = '2024-05-16'
#     initial_capital = 10000000000.0
#     state = simulate_trading(ticker, start_date, end_date, end_date_future, initial_capital)

#     return state
kst = timezone('Asia/Seoul')

# 초기 상태 설정
initial_capital = 10000000000.0
start_date = '2021-04-05'
end_date = '2024-05-17'
end_date_future = '2024-05-18'
state_avgo = None
state_aapl = None
state_asml = None
state_amzn = None
state_goog = None
state_nvda = None
state_meta = None
state_msft = None
state_cost = None
state_tsla = None
counter_avgo = 0
counter_aapl = 0
counter_asml = 0
counter_amzn = 0
counter_goog = 0
counter_nvda = 0
counter_meta = 0
counter_msft = 0
counter_cost = 0
counter_tsla = 0

def avgo_simulation():
    global state_avgo,counter_avgo,end_date,end_date_future,initial_capital,start_date
    saveNews('AVGO', end_date, end_date, "./test_AVGO/",False)  # googleNews활용
    state_avgo = simulate_trading('AVGO', start_date, end_date, end_date_future, initial_capital, counter_avgo)
    print(state_avgo,counter_avgo)
    with open('./test_AVGO.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_avgo)
    counter_avgo += 1
    if counter_avgo == 6:
        counter_avgo = 0
    print("AVGO 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_avgo)
#     return state_avgo,counter_avgo
    # return counter_avgo
def asml_simulation():
    
    global state_asml, counter_asml,end_date,end_date_future,initial_capital,start_date
    saveNews('ASML', end_date, end_date, "./test_ASML/",False)  # googleNews활용
    state_asml = simulate_trading('ASML', start_date, end_date, end_date_future, initial_capital, counter_asml)
    print(state_asml,counter_asml)
    with open('./test_ASML.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_asml)
    counter_asml += 1
    if counter_asml == 6:
        counter_asml = 0
    print("ASML 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_asml)
    
#     # return counter_asml
def aapl_simulation():
    
    global state_aapl, counter_aapl,end_date,end_date_future,initial_capital,start_date
    saveNews('AAPL', end_date, end_date, "./test_AAPL/",False)  # googleNews활용
    state_aapl = simulate_trading('AAPL', start_date, end_date, end_date_future, initial_capital, counter_aapl)
    print(state_aapl,counter_aapl)
    with open('./test_AAPL.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_aapl)
    counter_aapl += 1
    if counter_aapl == 6:
        counter_aapl = 0
    print("AAPL 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_aapl)
    # return counter_aapl
def amzn_simulation():
    
    global state_amzn, counter_amzn,end_date,end_date_future,initial_capital,start_date
    saveNews('AMZN', end_date, end_date, "./test_AMZN/",False)  # googleNews활용
    state_amzn = simulate_trading('AMZN', start_date, end_date, end_date_future, initial_capital, counter_amzn)
    counter_amzn += 1
    print(state_amzn,counter_amzn)
    with open('./test_AMZN.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_amzn)
    if counter_amzn == 6:
        counter_amzn = 0
    print("AMZN 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_amzn)
    return counter_amzn
def tsla_simulation():
    
    global state_tsla, counter_tsla,end_date,end_date_future,initial_capital,start_date
    saveNews('TSLA', end_date, end_date, "./test_TSLA/",False)  # googleNews활용
    state_tsla = simulate_trading('TSLA', start_date, end_date, end_date_future, initial_capital, counter_tsla)
    with open('./test_TSLA.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_tsla)
    counter_tsla += 1
    if counter_tsla == 6:
        counter_tsla = 0
    print("TSLA 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_tsla)
    return counter_tsla
def nvda_simulation():
    
    global state_nvda, counter_nvda,end_date,end_date_future,initial_capital,start_date
    saveNews('NVDA',end_date, end_date, "./test_NVDA/",False)  # googleNews활용
    state_nvda = simulate_trading('NVDA', start_date, end_date, end_date_future, initial_capital, counter_nvda)
    counter_nvda += 1
    with open('./test_NVDA.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_nvda)
    if counter_nvda == 6:
        counter_nvda = 0
    print("NVDA 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_nvda)
    return counter_nvda 
def meta_simulation():
    
    global state_meta, counter_meta,end_date,end_date_future,initial_capital,start_date
    saveNews('META', end_date, end_date, "./test_META/",False)  # googleNews활용
    state_meta = simulate_trading('META', start_date, end_date, end_date_future, initial_capital, counter_meta)
    counter_meta += 1
    with open('./test_META.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_meta)
    if counter_meta == 6:
        counter_meta = 0
    print("META 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_meta)
    return counter_meta
def cost_simulation():
   
    global state_cost, counter_cost,end_date,end_date_future,initial_capital,start_date
    saveNews('COST', end_date, end_date, "./test_COST/",False)  # googleNews활용
    state_cost = simulate_trading('COST', start_date, end_date, end_date_future, initial_capital, counter_cost)
    counter_cost += 1
    with open('./test_COST.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_cost)
    if counter_cost == 6:
        counter_cost = 0
    print("COST 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_cost)
    # return counter_cost
def msft_simulation():
    
    global state_msft, counter_msft,end_date,end_date_future,initial_capital,start_date
    saveNews('MSFT', end_date, end_date, "./test_MSFT/",False)  # googleNews활용
    state_msft = simulate_trading('MSFT', start_date, end_date, end_date_future, initial_capital, counter_cost)
    counter_msft += 1
    with open('./test_MSFT.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_msft)
    if counter_msft == 6:
        counter_msft = 0
    print("AVGO 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_msft)
    return counter_msft
def goog_simulation():
    
    global state_goog, counter_goog,end_date,end_date_future,initial_capital,start_date
    saveNews('GOOG', end_date, end_date, "./test_GOOG/",False)  # googleNews활용
    state_goog = simulate_trading('GOOG', start_date, end_date, end_date_future, initial_capital, counter_goog)
    print(state_goog,counter_goog)
    with open('./test_GOOG.txt', 'w', encoding='UTF-8') as stateFile:
        stateFile.write(state_goog)
    counter_goog += 1
    if counter_goog == 6:
        counter_goog = 0
    print("GOOG 시뮬레이션 실행 시각:", datetime.now(kst), "Counter:", counter_goog)
#     # return counter_goog
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# 백그라운드 스케줄러 생성
scheduler = BackgroundScheduler()

# 3:30부터 시작하여 4시까지 1시간 단위로 호출되도록 스케줄링
scheduler.add_job(lambda: executor.submit(avgo_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(asml_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(amzn_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(aapl_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(msft_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(meta_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(cost_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(goog_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(nvda_simulation), 'cron', hour='0,1', minute='2', timezone=kst)
scheduler.add_job(lambda: executor.submit(tsla_simulation), 'cron', hour='0,1', minute='2', timezone=kst)

# scheduler.add_job(lambda: executor.submit(tsla_simulation), 'cron', hour='22,23,0,1,2,3,4', minute='33', timezone=kst)

# scheduler.add_job(lambda: executor.submit(ag), 'cron', hour='20', minute='23', timezone=kst)
# counter = 1
# def ag():
#     global counter
#     counter +=1
#     print(counter)

# # 스케줄러 시작
scheduler.start()

# @app.get("/gg")
# async def agg():
#     global counter
#     return counter
# /avgo 호출시 미리 저장한 state를 반환합니다.
@app.get("/avgo")
async def avgo():
    # global state_avgo, counter_avgo, start_date, end_date, end_date_future, initial_capital
    # saveNews('AVGO', end_date, end_date, "./test_AVGO/",False)
    # state_avgo = simulate_trading('AVGO', start_date, end_date, end_date_future, initial_capital, counter_avgo)

    with open('./test_AVGO.txt', 'r', encoding='UTF-8') as stateFile:
        state_avgo = stateFile.readline()
    if state_avgo is None:
        return {"message": "작업이 아직 실행되지 않았습니다."}
    return {"state": "Buy"}

@app.get("/asml")
async def asml():
#     global state_asml, counter_asml, start_date, end_date, end_date_future, initial_capital
#     state_asml = simulate_trading('AVGO', start_date, end_date, end_date_future, initial_capital, counter_avgo)
    with open('./test_ASML.txt', 'r', encoding='UTF-8') as stateFile:
        state_asml = stateFile.readline()
    if state_asml is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_asml}
    return {"state": "Buy"}

@app.get("/aapl")
async def aapl():
    # global state_aapl,counter_aapl, start_date, end_date, end_date_future, initial_capital
    # state_aapl = simulate_trading('AAPL', start_date, end_date, end_date_future, initial_capital, counter_aapl)
    with open('./test_AAPL.txt', 'r', encoding='UTF-8') as stateFile:
        state_aapl = stateFile.readline()
    if state_aapl is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_aapl}
    return {"state": "Sell"}

@app.get("/tsla")
async def tsla():
    # global state_tsla,counter_tsla
    with open('./test_TSLA.txt', 'r', encoding='UTF-8') as stateFile:
        state_tsla = stateFile.readline()
    if state_tsla is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_tsla}
    return {"state": "Sell"}

@app.get("/amzn")
async def amzn():
    # global state_amzn,counter_amzn, start_date, end_date, end_date_future, initial_capital
    # saveNews('AMZN', end_date, end_date, "./test_AMZN/",False)  # googleNews활용

    # state_amzn = simulate_trading('AMZN', start_date, end_date, end_date_future, initial_capital, counter_amzn)
    with open('./test_AMZN.txt', 'r', encoding='UTF-8') as stateFile:
        state_amzn = stateFile.readline()
    if state_amzn is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_amzn}
    return {"state": "Sell"}

@app.get("/goog")
async def goog():
    # global state_goog,counter_goog, start_date, end_date, end_date_future, initial_capital
    # state_goog = simulate_trading('GOOG', start_date, end_date, end_date_future, initial_capital, counter_goog)
    with open('./test_GOOG.txt', 'r', encoding='UTF-8') as stateFile:
        state_goog = stateFile.readline()
    if state_goog is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_goog}
    return {"state": "Hold"}

@app.get("/cost")
async def cost():
    # global state_cost,counter_cost
    with open('./test_COST.txt', 'r', encoding='UTF-8') as stateFile:
        state_cost = stateFile.readline()
    if state_cost is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_cost}
    return {"state": "Hold"}

@app.get("/meta")
async def meta():
    # global state_meta,counter_meta
    with open('./test_META.txt', 'r', encoding='UTF-8') as stateFile:
        state_META = stateFile.readline()
    if state_meta is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_meta}
    return {"state": "Hold"}

@app.get("/nvda")
async def nvda():
    # global state_nvda,counter_nvda
    with open('./test_NVDA.txt', 'r', encoding='UTF-8') as stateFile:
        state_nvda = stateFile.readline()
    if state_nvda is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_nvda}
    return {"state": "Stay"}

@app.get("/msft")
async def msft():
    # global state_msft,counter_msft
    with open('./test_MSFT.txt', 'r', encoding='UTF-8') as stateFile:
        state_msft = stateFile.readline()
    if state_msft is None:
        return {"message": "작업이 아직 실행되지 않았습니다.","counter":counter_msft}
    return {"state": "Stay"}
# # 서버 종료시 스케줄러를 종료합니다.
# @app.on_event("shutdown")
# def shutdown_event():
#     scheduler.shutdown()
# !uvicorn main:app --host 0.0.0.0 --reload --port 8000






