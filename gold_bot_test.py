import numpy as np
from datetime import datetime, timedelta
import time

# ==================== تنظیمات تست ====================
INITIAL_CAPITAL = 100_000_000  # ریال
BUY_FEE = 0.005
SELL_FEE = 0.005
MIN_PROFIT_MARGIN = 0.005
SEQ_LEN = 10  # برای تست کوتاه‌تر کردیم

# داده‌های تست واقعی
test_prices = [
    136283000, 136350000, 136400000, 136380000, 136320000,
    136310000,  # ← سیگنال خرید باید اینجا بده
    136350000, 136420000, 136480000, 136550000,
    136620000, 136700000,  # ← سیگنال فروش باید اینجا بده
    136680000, 136650000, 136700000, 136750000,  # گزارش ساعتی
]

# =================================================

current_cash = INITIAL_CAPITAL
current_gold_grams = 0.0
has_bought_today = False
buy_price = 0
price_history_today = []

print("تست بات معاملاتی طلا شروع شد")
print(f"سرمایه اولیه: {current_cash:,} ریال\n")

for i, gold in enumerate(test_prices):
    now = datetime.now() + timedelta(minutes=i)
    price_history_today.append(gold)
    
    print(f"{now.strftime('%H:%M')} | قیمت طلا: {gold:,} ریال/گرم")

    # گزارش ساعتی (هر 60 دقیقه یکبار در تست)
    if i > 0 and i % 60 == 0:
        status = "دارای طلا" if has_bought_today else "نقدی"
        print(f"--- گزارش ساعتی ---")
        print(f"قیمت فعلی: {gold:,} ریال")
        print(f"وضعیت: {status} | سرمایه: {current_cash:,.0f} ریال\n")

    # سیگنال خرید
    if not has_bought_today and len(price_history_today) >= SEQ_LEN:
        # شبیه‌سازی پیش‌بینی ساده: میانگین + کمی کمتر از پایین‌ترین اخیر
        predicted_low = min(price_history_today[-SEQ_LEN:]) * 1.001
        if gold <= predicted_low * 1.005:
            buy_amount = current_cash / (1 + BUY_FEE)
            current_gold_grams = buy_amount / gold
            current_cash = 0
            buy_price = gold
            has_bought_today = True
            print(f"سیگنال خرید! خرید در {gold:,} ریال")
            print(f"مقدار طلا: {current_gold_grams:.6f} گرم\n")

    # سیگنال فروش
    if has_bought_today:
        gross = gold * current_gold_grams
        net = gross * (1 - SELL_FEE)
        cost = buy_price * current_gold_grams * (1 + BUY_FEE)
        profit_pct = (net - cost) / cost if cost > 0 else 0

        if profit_pct > MIN_PROFIT_MARGIN:
            profit_rial = net - cost
            current_cash = net
            print(f"سیگنال فروش! فروش در {gold:,} ریال")
            print(f"سود: {profit_pct:.2%} ({profit_rial:,.0f} ریال)")
            print(f"سرمایه جدید: {current_cash:,.0f} ریال\n")
            current_gold_grams = 0
            has_bought_today = False

print("تست تمام شد!")
print(f"سرمایه نهایی: {current_cash:,.0f} ریال")