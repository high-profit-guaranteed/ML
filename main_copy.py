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
app = FastAPI()

# @app.get('/tsla')
# def TSLA():
#     return {"data":"test"}

@app.get('/update')
def remoteUpdate():
    global data
    data = update()
    return "success"

# @app.get('/update')
async def update():
    ticker = 'AVGO'
    start_date = '2024-04-05'
    end_date = '2024-05-09'
    end_date_future = '2024-05-10'
    initial_capital = 10000000000.0
    state = await simulate_trading(ticker, start_date, end_date, end_date_future, initial_capital)
    return state

@app.get('/getData')
def getData():
    return data

@app.get('/avgo')
async def avgo():

    ticker = 'AVGO'
    start_date = '2021-04-05'
    end_date = '2024-05-15'
    end_date_future = '2024-05-16'
    initial_capital = 10000000000.0
    state = simulate_trading(ticker, start_date, end_date, end_date_future, initial_capital)

    return state
# !uvicorn main:app --host 0.0.0.0 --reload --port 8000