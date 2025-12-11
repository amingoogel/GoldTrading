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
API_KEY = "BQFbjkaU7Tdmr8zdtyq2k816GRi7RILR"
BASE_URL = "https://brsapi.ir/Api/Market/Gold_Currency.php"

# تنظیمات تلگرام — اینجا حتما پر کن
TELEGRAM_TOKEN = "8112942958:AAH8Hbg0cE7MRtzkg19isBZLLN08o0ikiqQ"   # ← توکن باتت
TELEGRAM_CHAT_ID = "1157963402"                                 # ← چت آیدی خودت

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}

BUY_FEE = 0.005
SELL_FEE = 0.005
MIN_PROFIT_MARGIN = 0.01
INITIAL_CAPITAL = 100000000  # ریال
CHECK_INTERVAL = 60
SEQ_LEN = 30
MODEL_PATH = "gold_brs_model.pth"
# =================================================

# ارسال پیام تلگرام
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "YOUR" in TELEGRAM_TOKEN:
        print("تلگرام تنظیم نشده →", text[:60] + "...")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except:
        pass  # اگر نت قطع بود، خطا نده

send_telegram_message("بات شروع به رصد کرد!")

# دریافت قیمت‌ها
def get_live_prices():
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params={"key": API_KEY}, timeout=15)
        if response.status_code != 200:
            return None, None, None
        data = response.json()

        gold_18 = usd = ons = 0

        # طلا ۱۸ عیار و انس
        for item in data.get('gold', []):
            name = (item.get('name') or '').lower()
            symbol = (item.get('symbol') or '').lower()
            price = item.get('price', 0)

            if '18' in name or '18 عیار' in name or '18k' in symbol:
                gold_18 = int(price) * 10  # تومان → ریال

            if 'انس' in name or 'xau' in symbol:
                ons = int(price)

        # دلار (اولویت: دلار آمریکا → تتر)
        for item in data.get('currency', []):
            name = (item.get('name') or '').lower()
            symbol = (item.get('symbol') or '').lower()
            price = item.get('price', 0)

            if 'دلار آمریکا' in name or 'usd_irt' in symbol or 'dollar usd' in symbol:
                usd = int(price) * 10
                break
        else:  # fallback به تتر
            for item in data.get('currency', []):
                if 'تتر' in (item.get('name') or '') or 'usdt' in (item.get('symbol') or '').lower():
                    usd = int(item.get('price', 0)) * 10
                    break

        return gold_18, usd, ons
    except:
        return None, None, None

# مدل و توابع یادگیری
class GoldDataset(Dataset):
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float32),
                torch.tensor(self.data[idx+self.seq_len, 0], dtype=torch.float32))

class CNNBiLSTM(nn.Module):
    def __init__(self, n_features=3, hidden_size=64):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bilstm1 = nn.LSTM(32, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.bilstm2 = nn.LSTM(hidden_size*2, hidden_size//2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        return self.fc(x[:, -1, :])

def train_model(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    dataset = GoldDataset(scaled)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CNNBiLSTM(n_features=df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(40):
        for seq, target in loader:
            out = model(seq)
            loss = criterion(out.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({"model": model.state_dict(), "scaler": scaler, "cols": df.shape[1]}, MODEL_PATH)
    print("مدل آموزش دید و ذخیره شد.")
    return model, scaler

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            ck = torch.load(MODEL_PATH, map_location='cpu')
            model = CNNBiLSTM(n_features=ck["cols"])
            model.load_state_dict(ck["model"])
            return model, ck["scaler"]
        except Exception as e:
            print("خطا در بارگذاری مدل:", e)
    return None, None

def predict_low_price(model, scaler, recent_data):
    if len(recent_data) < SEQ_LEN or model is None:
        return None
    recent = np.array(recent_data[-SEQ_LEN:])
    scaled = scaler.transform(recent)
    inp = torch.tensor(scaled).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(inp).item()
    dummy = np.zeros((1, recent.shape[1]))
    dummy[0, 0] = pred
    return scaler.inverse_transform(dummy)[0, 0]

# ==================== متغیرها و حلقه اصلی ====================
current_cash = INITIAL_CAPITAL
current_gold_grams = 0.0
has_bought_today = False
buy_price = 0
current_day = datetime.now().date()
price_history_today = []
model, scaler = load_model()
last_retrain = current_day

print(f"بات معاملاتی طلا + تلگرام فعال شد | سرمایه: {current_cash:,} ریال")

while True:
    now = datetime.now()

    # ریست روز
    if now.date() != current_day:
        current_day = now.date()
        has_bought_today = False
        buy_price = 0
        price_history_today = []
        print(f"\nروز جدید: {current_day}")

    # آموزش مجدد هفتگی
    if (current_day - last_retrain).days >= 7 and len(price_history_today) > 100:
        df = pd.DataFrame(price_history_today, columns=["gold", "usd", "ons"])
        model, scaler = train_model(df)
        last_retrain = current_day

    gold, usd, ons = get_live_prices()
    if gold is None or gold == 0:
        time.sleep(CHECK_INTERVAL)
        continue

    price_history_today.append([gold, usd or 0, ons or 0])
    print(f"{now.strftime('%H:%M:%S')} | طلا: {gold:,} ریال/گرم | دلار: {usd:,} | انس: {ons:,}")

    # سیگنال خرید
    if not has_bought_today and len(price_history_today) >= SEQ_LEN:
        predicted_low = predict_low_price(model, scaler, price_history_today)
        if predicted_low and gold <= predicted_low * 1.003:  # حاشیه 0.3%
            buy_amount = current_cash / (1 + BUY_FEE)
            current_gold_grams = buy_amount / gold
            current_cash = 0
            buy_price = gold
            has_bought_today = True

            msg = f"سیگنال خرید طلا\n" \
                  f"قیمت: {gold:,} ریال/گرم\n" \
                  f"پیش‌بینی پایین‌ترین: {predicted_low:,.0f}\n" \
                  f"مقدار: {current_gold_grams:.4f} گرم\n" \
                  f"{now.strftime('%H:%M %Y/%m/%d')} — برو میلی‌گلد خرید کن!"

            send_telegram_message(msg)
            print(msg)

    # سیگنال فروش
    if has_bought_today:
        gross = gold * current_gold_grams
        net = gross * (1 - SELL_FEE)
        cost = buy_price * current_gold_grams * (1 + BUY_FEE)
        profit_pct = (net - cost) / cost if cost > 0 else 0

        if profit_pct > MIN_PROFIT_MARGIN:
            profit_rial = net - cost
            current_cash = net

            msg = f"سیگنال فروش طلا\n" \
                  f"قیمت فروش: {gold:,} ریال/گرم\n" \
                  f"سود: {profit_pct:.2%} ({profit_rial:,.0f} ریال)\n" \
                  f"سرمایه جدید: {current_cash:,.0f} ریال\n" \
                  f"{now.strftime('%H:%M %Y/%m/%d')} — برو میلی‌گلد بفروش!"

            send_telegram_message(msg)
            print(msg)

            current_gold_grams = 0
            has_bought_today = False

    time.sleep(CHECK_INTERVAL)