import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
import os

# ==================== تنظیمات ====================
API_URL = "https://call2.tgju.org/ajax.json"

# تنظیمات تلگرام (حتما پر کن)
TELEGRAM_TOKEN = "8112942958:AAH8Hbg0cE7MRtzkg19isBZLLN08o0ikiqQ"
TELEGRAM_CHAT_ID = "1157963402"

BUY_FEE = 0.005
SELL_FEE = 0.005
MIN_PROFIT_MARGIN = 0.01
INITIAL_CAPITAL = 100000000  # ریال
CHECK_INTERVAL = 60
SEQ_LEN = 30
MODEL_PATH = "gold_tgju_model.pth"
# =================================================

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("تلگرام تنظیم نشده →", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except:
        pass

# دریافت قیمت طلا مستقیم از tgju API
send_telegram_message("start trading")

def get_live_prices():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()["current"]
            
            gold_keys = ["geram18", "tgju_gold_irg18", "tgju_gold_irg18_buy"]
            price_str = None
            
            for key in gold_keys:
                if key in data and "p" in data[key]:
                    price_str = data[key]["p"]
                    break
            
            if price_str is None:
                print("قیمت طلا پیدا نشد!")
                return None, None, None
            
            # حذف کاما
            price_rial = int(price_str.replace(",", ""))
            
            return price_rial, 0, 0  # مستقیم ریال برای هر گرم
        else:
            print(f"خطا در API: {response.status_code}")
            return None, None, None
    except Exception as e:
        print(f"خطا: {e}")
        return None, None, None

# ... بقیه کد بدون تغییر ...

# مدل ساده‌شده (فقط با قیمت طلا)
class GoldDataset(Dataset):
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32),
                torch.tensor(self.data[idx+self.seq_len], dtype=torch.float32))

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.unsqueeze(2)  # برای single feature
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

def train_model(prices):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    dataset = GoldDataset(scaled.flatten())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(30):
        for seq, target in loader:
            seq = seq.unsqueeze(0)
            out = model(seq)
            loss = criterion(out.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save({"model": model.state_dict(), "scaler": scaler}, MODEL_PATH)
    return model, scaler

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            ck = torch.load(MODEL_PATH, map_location='cpu')
            model = SimpleLSTM()
            model.load_state_dict(ck["model"])
            return model, ck["scaler"]
        except:
            pass
    return None, None

def predict_low_price(model, scaler, recent_prices):
    if len(recent_prices) < SEQ_LEN or model is None:
        return None
    recent = np.array(recent_prices[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(recent)
    inp = torch.tensor(scaled).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(inp).item()
    return scaler.inverse_transform([[pred]])[0][0]

# متغیرها
current_cash = INITIAL_CAPITAL
current_gold_grams = 0.0
has_bought_today = False
buy_price = 0
current_day = datetime.now().date()
price_history_today = []
model, scaler = load_model()
last_retrain = current_day

print("بات طلا با API tgju.org شروع شد | سرمایه اولیه:", current_cash, "ریال")

while True:
    now = datetime.now()

    if now.date() != current_day:
        current_day = now.date()
        has_bought_today = False
        buy_price = 0
        price_history_today = []
        print(f"\nروز جدید: {current_day}")

    if (current_day - last_retrain).days >= 7 and len(price_history_today) > 100:
        model, scaler = train_model(price_history_today)
        last_retrain = current_day

    gold, _, _ = get_live_prices()
    if gold is None or gold == 0:
        time.sleep(CHECK_INTERVAL)
        continue

    price_history_today.append(gold)
    print(f"{now.strftime('%H:%M:%S')} | طلا ۱۸ عیار: {gold:,} ریال/گرم")


    if not has_bought_today and len(price_history_today) >= SEQ_LEN:
        predicted_low = predict_low_price(model, scaler, price_history_today)
        if predicted_low and gold <= predicted_low * 1.003:
            buy_amount = current_cash / (1 + BUY_FEE)
            current_gold_grams = buy_amount / gold
            current_cash = 0
            buy_price = gold
            has_bought_today = True
            msg = f"🚨 سیگنال خرید!\nقیمت فعلی: {gold:,} ریال\nپیش‌بینی پایین‌ترین: {predicted_low:,.0f}\nمقدار: {current_gold_grams:.4f} گرم\nبرو میلی‌گلد خرید کن!"
            send_telegram_message(msg)
            print(msg)

    if has_bought_today:
        gross = gold * current_gold_grams
        net = gross * (1 - SELL_FEE)
        cost = buy_price * current_gold_grams * (1 + BUY_FEE)
        profit_pct = (net - cost) / cost if cost > 0 else 0
        if profit_pct > MIN_PROFIT_MARGIN:
            current_cash = net
            profit_rial = net - cost
            msg = f"🔔 سیگنال فروش!\nقیمت فروش: {gold:,} ریال\nسود: {profit_pct:.2%} ({profit_rial:,.0f} ریال)\nسرمایه جدید: {current_cash:,.0f} ریال\nبرو میلی‌گلد بفروش!"
            send_telegram_message(msg)
            print(msg)
            current_gold_grams = 0
            has_bought_today = False

    time.sleep(CHECK_INTERVAL)