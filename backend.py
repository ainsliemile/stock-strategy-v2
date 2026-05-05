from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import uvicorn

app = FastAPI()

# 設定 CORS，讓前端網頁可以順利連線
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    symbols: List[str]

def calculate_kd(df, n=9):
    df = df.copy()
    df['Lowest'] = df['Low'].rolling(window=n).min()
    df['Highest'] = df['High'].rolling(window=n).max()
    
    # 處理除以零的極端情況 (若 9 個月內最高價等於最低價)
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
    return df

def check_cross(line1, line2):
    # 確保資料長度足夠，避免報錯
    if len(line1) < 3 or len(line2) < 3:
        return "NONE"
    
    # 1. 檢查當下（包含未完結的當月/當週 K 棒）是否剛發生交叉
    current_golden = line1.iloc[-1] > line2.iloc[-1] and line1.iloc[-2] <= line2.iloc[-2]
    current_death = line1.iloc[-1] < line2.iloc[-1] and line1.iloc[-2] >= line2.iloc[-2]
    
    # 2. 檢查上一根（剛收盤的完整 K 棒）是否發生交叉，且目前趨勢未變 (防漏抓機制)
    prev_golden = line1.iloc[-2] > line2.iloc[-2] and line1.iloc[-3] <= line2.iloc[-3] and line1.iloc[-1] > line2.iloc[-1]
    prev_death = line1.iloc[-2] < line2.iloc[-2] and line1.iloc[-3] >= line2.iloc[-3] and line1.iloc[-1] < line2.iloc[-1]
    
    if current_golden or prev_golden:
        return "GOLDEN"
    if current_death or prev_death:
        return "DEATH"
        
    return "NONE"

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    results = []
    for symbol in request.symbols:
        try:
            ticker = yf.Ticker(symbol)
            # 抓取長天期資料以計算月線與週線
            hist = ticker.history(period="10y")
            if hist.empty:
                continue
                
            # === 取得週線資料並計算 MACD ===
            weekly_hist = hist.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            weekly_hist = calculate_macd(weekly_hist)
            weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
            
            # === 取得月線資料並計算 KD ===
            monthly_hist = hist.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            monthly_hist = calculate_kd(monthly_hist)
            
            # 確保資料長度足夠才進行判定
            if len(monthly_hist) < 3:
                continue
                
            monthly_k = round(monthly_hist['K'].iloc[-1], 1)
            monthly_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
            
            # === 買進策略判斷 (加入 K 值防呆機制) ===
            buy_signal = "觀望"
            
            # 取得「這個月」與「上一個月」的 K 值
            current_k = monthly_hist['K'].iloc[-1]
            prev_k = monthly_hist['K'].iloc[-2]
            
            # 只要現在 K < 30，就先給分批買進的資格
            if current_k < 30:
                buy_signal = "分批買進"
            
            # 嚴謹的大筆買進判定：
            if monthly_kd_cross == "GOLDEN":
                # 檢查發生交叉「當下」的那個月的 K 值是否真的有 < 30
                is_current_cross_valid = (monthly_hist['K'].iloc[-1] > monthly_hist['D'].iloc[-1]) and (current_k < 30)
                is_prev_cross_valid = (monthly_hist['K'].iloc[-2] > monthly_hist['D'].iloc[-2]) and (prev_k < 30)
                
                if is_current_cross_valid or is_prev_cross_valid:
                    buy_signal = "大筆買進"
                    
            # === 賣出策略判斷 ===
            sell_signal = "持有"
            if monthly_kd_cross == "DEATH":
                sell_signal = "全數賣出"
            elif weekly_macd_cross == "DEATH":
                sell_signal = "減碼50%"
                
            # 整理並回傳該檔股票的結果
            results.append({
                "symbol": symbol,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "monthly_k": monthly_k,
                "monthly_kd_cross": monthly_kd_cross,
                "weekly_macd_cross": weekly_macd_cross
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
            
    return results

if __name__ == "__main__":
    # 設定伺服器啟動的 Port
    uvicorn.run("backend:app", host="0.0.0.0", port=10000)
