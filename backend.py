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
    return {"status": "success", "message": "熊爸爸動能與波段 API 伺服器 (V2) 運作正常！🚀"}

class StockRequest(BaseModel):
    symbols: List[str]

# 這裡把手動設定 Session 的偽裝術刪除了，完全交給 yfinance 原廠的防擋機制來處理！

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
    if current_golden or prev_golden:
        return "GOLDEN"
    if current_death or prev_death:
        return "DEATH"
    return "NONE"

def check_bullish_divergence(df):
    if len(df) < 5:
        return False
    recent_closes = df['Close'].iloc[-5:]
    recent_hists = df['Hist'].iloc[-5:]
    min_close_idx = recent_closes.idxmin()
    min_hist_idx = recent_hists.idxmin()
    if min_hist_idx < min_close_idx and df['Hist'].iloc[-1] > df['Hist'].loc[min_hist_idx] and df['Hist'].iloc[-1] < 0:
        return True
    return False

def fetch_data_with_retry(symbol, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.4))
            # 讓 yfinance 自己處理連線，不再強制塞入 session
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10y")
            if not hist.empty:
                return hist
        except Exception as e:
            print(f"⚠️ [{symbol}] 抓取失敗 (嘗試 {attempt + 1}/{retries}): {e}", flush=True)
            time.sleep(1)
    return pd.DataFrame()

def process_symbol(symbol):
    try:
        hist = fetch_data_with_retry(symbol)
        
        if hist.empty:
            print(f"❌ [{symbol}] 徹底失敗：無資料或 IP 遭阻擋", flush=True)
            return {
                "symbol": symbol, "buy_signal": "無資料/IP遭擋", "sell_signal": "-",
                "monthly_k": 0.0, "monthly_kd_cross": "-", "weekly_macd_cross": "-", "momentum_score": -999.0
            }
            
        current_bias = 0.0
        if len(hist) >= 240:
            hist['MA240'] = hist['Close'].rolling(window=240).mean()
            current_bias = clean_float((hist['Close'].iloc[-1] - hist['MA240'].iloc[-1]) / hist['MA240'].iloc[-1])
            
        agg_dict = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        weekly_hist = hist.resample('W').agg(agg_dict).dropna()
        weekly_hist = calculate_macd(weekly_hist)
        weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
        weekly_divergence = check_bullish_divergence(weekly_hist)
        
        try:
            monthly_hist = hist.resample('ME').agg(agg_dict).dropna()
        except ValueError:
            monthly_hist = hist.resample('M').agg(agg_dict).dropna()
            
        monthly_hist = calculate_kd(monthly_hist)
        
        if len(monthly_hist) < 5:
             print(f"⚠️ [{symbol}] 資料太短：上市不到 5 個月", flush=True)
             return {
                "symbol": symbol, "buy_signal": "上市時間太短", "sell_signal": "-",
                "monthly_k": 0.0, "monthly_kd_cross": "-", "weekly_macd_cross": "-", "momentum_score": -999.0
            }
            
        monthly_hist['MA5'] = monthly_hist['Close'].rolling(window=5).mean()
        monthly_hist['Vol5'] = monthly_hist['Volume'].rolling(window=5).mean()
        
        monthly_k = round(clean_float(monthly_hist['K'].iloc[-1]), 1)
        monthly_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
        
        current_k = clean_float(monthly_hist['K'].iloc[-1])
        prev_k = clean_float(monthly_hist['K'].iloc[-2])
        current_close = clean_float(monthly_hist['Close'].iloc[-1])
        current_ma5 = clean_float(monthly_hist['MA5'].iloc[-1])
        current_vol = clean_float(monthly_hist['Volume'].iloc[-1])
        avg_vol5 = clean_float(monthly_hist['Vol5'].iloc[-1])
        
        buy_signal = "觀望"
        is_golden = (monthly_kd_cross == "GOLDEN") or (current_k > clean_float(monthly_hist['D'].iloc[-1]) and prev_k <= clean_float(monthly_hist['D'].iloc[-2]))
        
        if is_golden and current_k < 40 and current_close > current_ma5 and current_vol > avg_vol5:
            buy_signal = "右側重壓(50%)"
        elif current_k < 20 and weekly_divergence:
            buy_signal = "底部加碼(30%)"
        elif current_k < 30 and current_bias < -0.15:
            buy_signal = "左側建倉(20%)"
            
        sell_signal = "持有"
        if monthly_kd_cross == "DEATH":
            sell_signal = "全數賣出"
        elif weekly_macd_cross == "DEATH":
            sell_signal = "減碼50%"
            
        momentum_score = -999.0 
        if len(monthly_hist) >= 7: 
            ret_1m = clean_float(current_close / clean_float(monthly_hist['Close'].iloc[-2])) - 1
            ret_3m = clean_float(current_close / clean_float(monthly_hist['Close'].iloc[-4])) - 1
            ret_6m = clean_float(current_close / clean_float(monthly_hist['Close'].iloc[-7])) - 1
            momentum_score = clean_float((ret_1m + ret_3m + ret_6m) / 3)
            
        print(f"✅ [{symbol}] 處理完成！動能: {momentum_score:.2f}, 買進燈號: {buy_signal}", flush=True)
        return {
            "symbol": symbol, "buy_signal": buy_signal, "sell_signal": sell_signal,
            "monthly_k": monthly_k, "monthly_kd_cross": monthly_kd_cross,
            "weekly_macd_cross": weekly_macd_cross, "momentum_score": momentum_score 
        }
    except Exception as e:
        print(f"💣 [{symbol}] 發生嚴重錯誤崩潰: {e}", flush=True)
        return {
            "symbol": symbol, "buy_signal": "運算錯誤", "sell_signal": "-",
            "monthly_k": 0.0, "monthly_kd_cross": "-", "weekly_macd_cross": "-", "momentum_score": -999.0
        }

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    print(f"\n==============================================", flush=True)
    print(f"🟢 收到前端分析請求！共有 {len(request.symbols)} 檔標的等待運算...", flush=True)
    start_time = time.time()
    
    raw_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in request.symbols}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                raw_results.append(res)
                
    raw_results.sort(key=lambda x: x['momentum_score'], reverse=True)
    final_results = []
    current_rank = 1
    
    for item in raw_results:
        score = item['momentum_score']
        if score == -999.0:
            item['momentum_rank'] = "資料不足"
            item['momentum_score_str'] = "-"
        elif score < 0:
            item['momentum_rank'] = "不投資" 
            item['momentum_score_str'] = f"{score * 100:.2f}%"
        else:
            item['momentum_rank'] = f"第 {current_rank} 名"
            item['momentum_score_str'] = f"{score * 100:.2f}%"
            current_rank += 1
        final_results.append(item)
            
    end_time = time.time()
    print(f"🏁 全部運算結束！共花費 {end_time - start_time:.2f} 秒。成功傳回 {len(final_results)} 筆結果。\n==============================================", flush=True)
    return final_results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
