from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import uvicorn
import os

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
    # 增加 MACD 柱狀圖計算，用於判斷背離
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

# 偵測週線底背離 (Bullish Divergence)
def check_bullish_divergence(df):
    if len(df) < 5: 
        return False
    # 觀察近 5 週的價格與 MACD 柱狀圖
    recent_closes = df['Close'].iloc[-5:]
    recent_hists = df['Hist'].iloc[-5:]
    
    min_close_idx = recent_closes.idxmin()
    min_hist_idx = recent_hists.idxmin()
    
    # 如果 MACD 最低綠柱發生在價格最低點的「前面」(殺盤動能提早縮減)
    # 且目前的柱狀圖正在收斂(上升)且仍處於水下
    if min_hist_idx < min_close_idx and df['Hist'].iloc[-1] > df['Hist'].loc[min_hist_idx] and df['Hist'].iloc[-1] < 0:
        return True
    return False

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    raw_results = []
    
    for symbol in request.symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10y")
            if hist.empty:
                continue
            
            # --- 計算年線 (240日) 乖離率 ---
            current_bias = 0
            if len(hist) >= 240:
                hist['MA240'] = hist['Close'].rolling(window=240).mean()
                current_bias = (hist['Close'].iloc[-1] - hist['MA240'].iloc[-1]) / hist['MA240'].iloc[-1]
                
            # === 取得週線資料並計算 MACD 與底背離 ===
            # 注意：加上 'Volume':'sum' 以計算均量
            weekly_hist = hist.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            weekly_hist = calculate_macd(weekly_hist)
            weekly_macd_cross = check_cross(weekly_hist['MACD'], weekly_hist['Signal'])
            weekly_divergence = check_bullish_divergence(weekly_hist)
            
            # === 取得月線資料並計算 KD 與均線/均量 ===
            monthly_hist = hist.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            monthly_hist = calculate_kd(monthly_hist)
            
            if len(monthly_hist) < 5:
                continue
                
            # 計算月線的 5月均線 與 5月均量 (用於右側確認)
            monthly_hist['MA5'] = monthly_hist['Close'].rolling(window=5).mean()
            monthly_hist['Vol5'] = monthly_hist['Volume'].rolling(window=5).mean()
            
            monthly_k = round(monthly_hist['K'].iloc[-1], 1)
            monthly_kd_cross = check_cross(monthly_hist['K'], monthly_hist['D'])
            
            current_k = monthly_hist['K'].iloc[-1]
            prev_k = monthly_hist['K'].iloc[-2]
            current_close = monthly_hist['Close'].iloc[-1]
            current_ma5 = monthly_hist['MA5'].iloc[-1]
            current_vol = monthly_hist['Volume'].iloc[-1]
            avg_vol5 = monthly_hist['Vol5'].iloc[-1]
            
            # === 買進策略判斷 (三階段智慧建倉) ===
            buy_signal = "觀望"
            
            # 判斷是否為有效金叉
            is_golden = (monthly_kd_cross == "GOLDEN") or (monthly_hist['K'].iloc[-1] > monthly_hist['D'].iloc[-1] and prev_k <= monthly_hist['D'].iloc[-2])
            
            # 階段三：右側重壓 (月 KD 金叉 且 放量站上 5月均線)
            if is_golden and current_k < 40 and current_close > current_ma5 and current_vol > avg_vol5:
                buy_signal = "右側重壓(50%)"
                    
            # 階段二：底部加碼 (月 K < 20 且 出現週線級別底背離)
            elif current_k < 20 and weekly_divergence:
                buy_signal = "底部加碼(30%)"
                
            # 階段一：左側建倉 (月 K < 30 且 距年線乖離率 < -15%)
            elif current_k < 30 and current_bias < -0.15:
                buy_signal = "左側建倉(20%)"
                    
            # === 賣出策略判斷 ===
            sell_signal = "持有"
            if monthly_kd_cross == "DEATH":
                sell_signal = "全數賣出"
            elif weekly_macd_cross == "DEATH":
                sell_signal = "減碼50%"
                
            # --- Papa Bear 動能指標計算 (1M, 3M, 6M 平均) ---
            momentum_score = -999 
            if len(monthly_hist) >= 7: 
                current_close = monthly_hist['Close'].iloc[-1]
                ret_1m = (current_close / monthly_hist['Close'].iloc[-2]) - 1
                ret_3m = (current_close / monthly_hist['Close'].iloc[-4]) - 1
                ret_6m = (current_close / monthly_hist['Close'].iloc[-7]) - 1
                momentum_score = (ret_1m + ret_3m + ret_6m) / 3
                
            raw_results.append({
                "symbol": symbol,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "monthly_k": monthly_k,
                "monthly_kd_cross": monthly_kd_cross,
                "weekly_macd_cross": weekly_macd_cross,
                "momentum_score": momentum_score 
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
            
    # --- 執行排名系統 ---
    raw_results.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    final_results = []
    current_rank = 1
    
    for item in raw_results:
        score = item['momentum_score']
        if score == -999:
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
            
    return final_results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
