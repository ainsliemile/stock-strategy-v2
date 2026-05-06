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

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    raw_results = []
    
    for symbol in request.symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10y")
            if hist.empty:
                continue
                
            weekly_hist = hist.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            weekly_hist = calculate_macd(weekly_hist)
            weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
            
            monthly_hist = hist.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            monthly_hist = calculate_kd(monthly_hist)
            
            if len(monthly_hist) < 3:
                continue
                
            monthly_k = round(monthly_hist['K'].iloc[-1], 1)
            monthly_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
            
            # --- 買賣訊號判斷 ---
            buy_signal = "觀望"
            current_k = monthly_hist['K'].iloc[-1]
            prev_k = monthly_hist['K'].iloc[-2]
            
            if current_k < 30:
                buy_signal = "分批買進"
            
            if monthly_kd_cross == "GOLDEN":
                is_current_cross_valid = (monthly_hist['K'].iloc[-1] > monthly_hist['D'].iloc[-1]) and (current_k < 30)
                is_prev_cross_valid = (monthly_hist['K'].iloc[-2] > monthly_hist['D'].iloc[-2]) and (prev_k < 30)
                if is_current_cross_valid or is_prev_cross_valid:
                    buy_signal = "大筆買進"
                    
            sell_signal = "持有"
            if monthly_kd_cross == "DEATH":
                sell_signal = "全數賣出"
            elif weekly_macd_cross == "DEATH":
                sell_signal = "減碼50%"
                
            # --- Papa Bear 動能指標計算 (3M, 6M, 12M 平均) ---
            momentum_score = -999 # 預設為無效值
            if len(monthly_hist) >= 13: # 確保有超過一年的資料
                current_close = monthly_hist['Close'].iloc[-1]
                # 回溯 3 個月、6 個月、12 個月的收盤價計算報酬率
                ret_3m = (current_close / monthly_hist['Close'].iloc[-4]) - 1
                ret_6m = (current_close / monthly_hist['Close'].iloc[-7]) - 1
                ret_12m = (current_close / monthly_hist['Close'].iloc[-13]) - 1
                momentum_score = (ret_3m + ret_6m + ret_12m) / 3
                
            raw_results.append({
                "symbol": symbol,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "monthly_k": monthly_k,
                "monthly_kd_cross": monthly_kd_cross,
                "weekly_macd_cross": weekly_macd_cross,
                "momentum_score": momentum_score # 儲存未格式化的原始數值供排序用
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
            
    # --- 執行排名系統 ----
    # 先依照動能分數由高到低排序
    raw_results.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    final_results = []
    current_rank = 1
    
    for item in raw_results:
        score = item['momentum_score']
        
        # 格式化數值與判定排名
        if score == -999:
            item['momentum_rank'] = "資料不足"
            item['momentum_score_str'] = "-"
        elif score < 0:
            item['momentum_rank'] = "不投資" # 客製化條件：小於0顯示不投資
            item['momentum_score_str'] = f"{score * 100:.2f}%"
        else:
            item['momentum_rank'] = f"第 {current_rank} 名"
            item['momentum_score_str'] = f"{score * 100:.2f}%"
            current_rank += 1 # 只有大於0的才給予實際排名並遞增
            
        final_results.append(item)
            
    return final_results

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=10000)
