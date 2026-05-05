from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import uvicorn

app = FastAPI()

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
    df['RSV'] = 100 * (df['Close'] - df['Lowest']) / (df['Highest'] - df['Lowest'])
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
    if len(line1) < 2 or len(line2) < 2:
        return "NONE"
    if line1.iloc[-1] > line2.iloc[-1] and line1.iloc[-2] <= line2.iloc[-2]:
        return "GOLDEN"
    if line1.iloc[-1] < line2.iloc[-1] and line1.iloc[-2] >= line2.iloc[-2]:
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
                
            # 轉換為週線並計算 MACD
            weekly_hist = hist.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            weekly_hist = calculate_macd(weekly_hist)
            weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
            
            # 轉換為月線並計算 KD
            monthly_hist = hist.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            monthly_hist = calculate_kd(monthly_hist)
            monthly_k = round(monthly_hist['K'].iloc[-1], 1)
            monthly_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
            
            # === 買進策略判斷 ===
            buy_signal = "觀望"
            if monthly_k < 30:
                if monthly_kd_cross == "GOLDEN":
                    buy_signal = "大筆買進"
                else:
                    buy_signal = "分批買進"
                    
            # === 賣出策略判斷 ===
            sell_signal = "持有"
            if monthly_kd_cross == "DEATH":
                sell_signal = "全數賣出"
            elif weekly_macd_cross == "DEATH":
                sell_signal = "減碼50%"
                
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
    uvicorn.run("backend:app", host="0.0.0.0", port=10000)