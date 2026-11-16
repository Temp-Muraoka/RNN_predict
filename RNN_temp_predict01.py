"""
RNNを用いた時系列データ予測

出力をOUTPUT_WINDOWで定義して、
指定個数の未来を予測させる。

RNN,GRU,LSTM
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. データのロード（キャッシュ対応 ＆ 失敗時終了） ---
csv_path = "tokyo_weather.csv"

if os.path.exists(csv_path):
    print("ローカルファイルから気象データを読み込みます。")
    df = pd.read_csv(csv_path)
else:
    print("APIから気象データをダウンロードしています...")
    # 気温、湿度、気圧を含むOpen-Meteoのエンドポイント
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=35.6895&longitude=139.6917&start_date=2021-01-01&end_date=2025-12-31&daily=temperature_2m_max,relative_humidity_2m_max,surface_pressure_mean"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # エラー時に例外を発生させる
        data = response.json()
        
        df = pd.DataFrame({
            'date': data['daily']['time'],
            'temperature': data['daily']['temperature_2m_max'],
            'humidity': data['daily']['relative_humidity_2m_max'],
            'pressure': data['daily']['surface_pressure_mean']
        })
        
        # 欠損値の補完
        df = df.ffill().bfill()
        
        # ローカルに保存
        df.to_csv(csv_path, index=False)
        print("データをダウンロードし、ローカルに保存しました。")
        
    except Exception as e:
        print(f"エラー: データの取得に失敗しました ({e})。プログラムを終了します。")
        sys.exit(1)

# --- 2. ハイパーパラメータの設定 ---
INPUT_WINDOW = 30   # 過去何日分のデータを見るか（入力）
OUTPUT_WINDOW = 10  # 未来何日分を直接予測するか（最後の10個と比較するため10に設定）
HIDDEN_SIZE = 6    # 隠れ層の次元数
NUM_LAYERS = 1      # RNNレイヤー数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 3. 最新10個を予測対象（未来データ）として完全に切り離す ---
future_df = df.tail(10).copy()      # 最後の10日分（学習には絶対に使わない）
train_full_df = df.iloc[:-10].copy()  # 残りのデータで学習・テストを行う

# --- 4. 前処理とスケーリング ---
# 説明変数（特徴量）として「気温」「湿度」「気圧」の3つを指定
feature_cols = ['temperature', 'humidity', 'pressure']

# 特徴量全体のスケーラー
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(train_full_df[feature_cols].values)

# 予測対象（気温）専用の逆スケーラー
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(train_full_df[['temperature']])

# スライディングウィンドウ作成関数
def create_dataset(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i : i + input_window, :])        # 入力: 全特徴量 (input_window, 3)
        y.append(data[i + input_window : i + input_window + output_window, 0]) # 出力: 気温のみ (output_window,)
    return np.array(X), np.array(y)

X_all, y_all = create_dataset(scaled_features, INPUT_WINDOW, OUTPUT_WINDOW)

# 訓練データとテストデータの分割 (80% / 20%)
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# PyTorch Datasetの定義
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WeatherDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- 5. RNNモデルの定義（入力次元 = 3） ---
class DirectRNNPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_window=10, num_layers=1):
        super(DirectRNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM使用時はforwardも変更要
        self.rnn = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_window)
        
    def forward(self, x):
        
        # # RNN,GRU
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # out, _ = self.rnn(x, h0) # 出力、隠れ状態

        # LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    
        out, (hn, cn) = self.rnn(x, (h0, c0)) # タプルで受け取る

        
        last_hidden = out[:, -1, :]
        out_pred = self.fc(last_hidden)
        return out_pred

# --- 6. 学習の実行 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

model = DirectRNNPredictor(
    input_size=len(feature_cols), 
    hidden_size=HIDDEN_SIZE, 
    output_window=OUTPUT_WINDOW, 
    num_layers=NUM_LAYERS
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_history = []
test_loss_history = []

for epoch in range(EPOCHS):
    # 訓練フェーズ
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_X).squeeze(-1)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)

    # テスト（評価）フェーズ
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze(-1)
            loss = criterion(preds, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss = test_loss / len(test_loader.dataset)
    test_loss_history.append(test_loss)

    print(f"エポック [{epoch+1}/{EPOCHS}] | 訓練Loss: {train_loss:.5f} | テストLoss: {test_loss:.5f}")

print("学習が完了しました！\n")

# --- 学習曲線 ---
plt.figure(figsize=(6, 4))

# 1. 学習曲線のプロット
plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss (MSE)', color='blue')
plt.plot(range(1, EPOCHS + 1), test_loss_history, label='Test Loss (MSE)', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves (Train vs Test Loss)')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. 学習完了後：最後の10個の予測と実際のデータの比較 ---
model.eval()
with torch.no_grad():
    # 最後の10個を予測するため、その直前の「INPUT_WINDOW（30日分）」のデータを取得
    recent_data_window = train_full_df.tail(INPUT_WINDOW)[feature_cols].values
    scaled_recent = scaler_x.transform(recent_data_window)
    
    # テンソルに変換してモデルに入力 (形状: [1, 30, 3])
    input_tensor = torch.tensor(scaled_recent, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 10日分の未来を直接予測
    pred_scaled = model(input_tensor).cpu().numpy().squeeze(0)
    
    # 予測値を元のスケール（実際の気温の単位）に戻す
    pred_temp = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# 実測値と日付の取得
actual_temp = future_df['temperature'].values
dates = future_df['date'].values

# 表形式で比較を表示
comparison_df = pd.DataFrame({
    '日付': dates,
    '実測値 (Actual)': actual_temp,
    '予測値 (Predicted)': pred_temp
})
print("--- 最後の10日間の予測結果と実際のデータの比較 ---")
print(comparison_df.to_string(index=False))

# グラフによる比較の可視化
plt.figure(figsize=(10, 5))
plt.plot(dates, actual_temp, marker='o', label='Actual Temperature', color='blue')
plt.plot(dates, pred_temp, marker='x', linestyle='--', label='Predicted Temperature', color='orange')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Comparison of Actual vs Predicted Temperature (Last 10 Days)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()