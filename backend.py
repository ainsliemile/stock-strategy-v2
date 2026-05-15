import os
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import uvicorn
import concurrent.futures
import time
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "success", "message": "熊爸爸數據透視 API 伺服器運作正常！🚀"}

class StockRequest(BaseModel):
    symbols: List[str]

def clean_float(val):
    if pd.isna(val) or math.isnan(val) or math.isinf(val):
        return 0.0
    return float(val)

def calculate_kd(df, n=9):
    df = df.copy()
    df['Lowest'] = df['Low'].rolling(window=n).min()
    df['Highest'] = df['High'].rolling(window=n).max()
    denominator = df['Highest'] - df['Lowest']
    df['RSV'] = 100 * (df['Close'] - df['Lowest']) / denominator.replace(0, 1)
    df['RSV'] = df['RSV'].fillna(50)
    K, D = [50], [50]
    for rsv in df['RSV'].iloc[1:]:
        k_val = (2/3) * K[-1] + (1/3) * rsv
        d_val = (2/3) * D[-1] + (1/3) * k_val
        K.append(k_val)
        D.append(d_val)
    df['K'] = K
    df['D'] = D
    return df

def calculate_macd(df):
    df = df.copy()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def check_cross(line1, line2):
    if len(line1) < 3 or len(line2) < 3:
        return "NONE"
    current_golden = line1.iloc[-1] > line2.iloc[-1] and line1.iloc[-2] <= line2.iloc[-2]
    current_death = line1.iloc[-1] < line2.iloc[-1] and line1.iloc[-2] >= line2.iloc[-2]
    prev_golden = line1.iloc[-2] > line2.iloc[-2] and line1.iloc[-3] <= line2.iloc[-3] and line1.iloc[-1] > line2.iloc[-1]
    prev_death = line1.iloc[-2] < line2.iloc[-2] and line1.iloc[-3] >= line2.iloc[-3] and line1.iloc[-1] < line2.iloc[-1]
    if current_golden or prev_golden: return "GOLDEN"
    if current_death or prev_death: return "DEATH"
    return "NONE"

def check_bullish_divergence(df):
    if len(df) < 5: return False
    recent_closes = df['Close'].iloc[-5:]
    recent_hists = df['Hist'].iloc[-5:]
    min_close_idx = recent_closes.idxmin()
    min_hist_idx = recent_hists.idxmin()
    if min_hist_idx < min_close_idx and df['Hist'].iloc[-1] > df['Hist'].loc[min_hist_idx] and df['Hist'].iloc[-1] < 0:
        return True
    return False

def fetch_data_with_retry(symbol, retries=2):
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10y")
            if not hist.empty: return hist
        except:
            time.sleep(1)
    return pd.DataFrame()

def process_symbol(symbol):
    try:
        hist = fetch_data_with_retry(symbol)
        if hist.empty:
            return {"symbol": symbol, "buy_signal": "無資料", "sell_signal": "-", "momentum_score": -999.0}
            
        # 取得當前股價 (最後一筆收盤價)
        current_price = round(clean_float(hist['Close'].iloc[-1]), 2)
            
        current_bias = 0.0
        if len(hist) >= 240:
            ma240 = hist['Close'].rolling(window=240).mean().iloc[-1]
            current_bias = (hist['Close'].iloc[-1] - ma240) / ma240
            
        weekly_hist = hist.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        weekly_hist = calculate_macd(weekly_hist)
        weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
        weekly_divergence = check_bullish_divergence(weekly_hist)
        
        # 20週線與 MACD 柱狀體狀態判斷
        current_w_ma20 = 0.0
        if len(weekly_hist) >= 20:
            weekly_hist['MA20'] = weekly_hist['Close'].rolling(window=20).mean()
            current_w_ma20 = clean_float(weekly_hist['MA20'].iloc[-1])
            
        weekly_macd_hist_val = clean_float(weekly_hist['Hist'].iloc[-1])
        weekly_hist_shrinking = False
        if len(weekly_hist) >= 3:
            h1 = clean_float(weekly_hist['Hist'].iloc[-1])
            h2 = clean_float(weekly_hist['Hist'].iloc[-2])
            h3 = clean_float(weekly_hist['Hist'].iloc[-3])
            if h3 > 0 and h2 < h3 and h1 < h2:
                weekly_hist_shrinking = True
                
        current_weekly_close = clean_float(weekly_hist['Close'].iloc[-1])
        
        try:
            monthly_hist = hist.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        except:
            monthly_hist = hist.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            
        monthly_hist = calculate_kd(monthly_hist)
        monthly_hist['MA5'] = monthly_hist['Close'].rolling(window=5).mean()
        monthly_hist['Vol5'] = monthly_hist['Volume'].rolling(window=5).mean()
        
        m_k = round(clean_float(monthly_hist['K'].iloc[-1]), 1)
        m_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
        
        current_k = clean_float(monthly_hist['K'].iloc[-1])
        prev_k = clean_float(monthly_hist['K'].iloc[-2])
        current_d = clean_float(monthly_hist['D'].iloc[-1])
        prev_d = clean_float(monthly_hist['D'].iloc[-2])
        
        is_golden = (m_kd_cross == "GOLDEN") or (current_k > current_d and prev_k <= prev_d)
        is_m_death = (m_kd_cross == "DEATH") or (current_k < current_d and prev_k >= prev_d)
        
        buy_signal = "觀望"
        if is_golden and m_k < 40 and float(monthly_hist['Close'].iloc[-1]) > float(monthly_hist['MA5'].iloc[-1]) and float(monthly_hist['Volume'].iloc[-1]) > float(monthly_hist['Vol5'].iloc[-1]):
            buy_signal = "右側重壓(50%)"
        elif m_k < 20 and weekly_divergence:
            buy_signal = "底部加碼(30%)"
        elif m_k < 30 and current_bias < -0.15:
            buy_signal = "左側建倉(20%)"
            
        sell_signal = "持有"
        if is_m_death:
            if current_w_ma20 > 0 and current_weekly_close < current_w_ma20:
                sell_signal = "清倉(100%)"
            else:
                sell_signal = "緊急減碼(50%)"
        elif weekly_macd_cross == "DEATH":
            sell_signal = "頂部減碼(30%)"
        elif weekly_hist_shrinking:
            sell_signal = "預警減碼(20%)"
            
        momentum_score = -999.0 
        if len(monthly_hist) >= 7: 
            ret_1m = (monthly_hist['Close'].iloc[-1] / monthly_hist['Close'].iloc[-2]) - 1
            ret_3m = (monthly_hist['Close'].iloc[-1] / monthly_hist['Close'].iloc[-4]) - 1
            ret_6m = (monthly_hist['Close'].iloc[-1] / monthly_hist['Close'].iloc[-7]) - 1
            momentum_score = float((ret_1m + ret_3m + ret_6m) / 3)
            
        return {
            "symbol": symbol,
            "current_price": current_price,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "monthly_k": m_k,
            "monthly_kd_cross": m_kd_cross,
            "weekly_macd_cross": weekly_macd_cross,
            "weekly_macd_hist": round(weekly_macd_hist_val, 2),
            "weekly_hist_shrinking": weekly_hist_shrinking,
            "weekly_ma20": round(current_w_ma20, 2),
            "weekly_divergence": weekly_divergence,
            "bias_val": round(current_bias * 100, 2),
            "momentum_score": momentum_score 
        }
    except Exception as e:
        return {"symbol": symbol, "buy_signal": "運算錯誤", "momentum_score": -999.0}

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    raw_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in request.symbols}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: raw_results.append(res)
                
    raw_results.sort(key=lambda x: x.get('momentum_score', -999.0), reverse=True)
    final_results = []
    current_rank = 1
    for item in raw_results:
        score = item.get('momentum_score', -999.0)
        if score == -999.0: item['momentum_rank'], item['momentum_score_str'] = "資料不足", "-"
        elif score < 0: item['momentum_rank'], item['momentum_score_str'] = "不投資", f"{score * 100:.2f}%"
        else:
            item['momentum_rank'] = f"第 {current_rank} 名"
            item['momentum_score_str'] = f"{score * 100:.2f}%"
            current_rank += 1
        final_results.append(item)
    return final_results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
