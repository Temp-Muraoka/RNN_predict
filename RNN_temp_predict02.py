"""
RNNを用いた時系列データ予測

出力の時間はt+1のみ
長期間予測には自己回帰で求める

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

# --- データのロード（キャッシュ対応 ＆ 失敗時終了） ---
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

# --- ハイパーパラメータの設定 ---
INPUT_WINDOW = 30   # 過去何日分のデータを見るか（入力）
OUTPUT_WINDOW = 1   # 未来「1つだけ」を予測対象にする
HIDDEN_SIZE = 6    # 隠れ層の次元数
NUM_LAYERS = 1      # RNNレイヤー数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# --- 前処理とスケーリング ---
# 説明変数（特徴量）として「気温」「湿度」「気圧」の3つを指定
feature_cols = ['temperature', 'humidity', 'pressure']

# 特徴量全体のスケーラー
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(df[feature_cols].values)

# 予測対象（気温）専用の逆スケーラー
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(df[['temperature']])

# スライディングウィンドウ作成関数
def create_dataset(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i : i + input_window, :])        # 入力: 全特徴量 (input_window, 3)
        y.append(data[i + input_window : i + input_window + output_window, 0]) # 出力: 気温のみ (output_window,)
    return np.array(X), np.array(y)

X_all, y_all = create_dataset(scaled_features, INPUT_WINDOW, OUTPUT_WINDOW)
y_all = y_all.squeeze(-1) # 形状調整 (samples,)

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

# # --- テストデータ全体に対する「自己回帰（連続）予測」の実行 ---
# model.eval()
# with torch.no_grad():
#     # テストデータの最初の入力ウィンドウを取得
#     # （テスト区間の開始点よりINPUT_WINDOW分さかのぼったところからスタート）
#     start_idx = train_size
#     current_window = scaled_features[start_idx : start_idx + INPUT_WINDOW].copy()
    
#     predictions_scaled = []
#     # テストデータのサンプル数分だけ1ステップ予測を繰り返す
#     for i in range(len(X_test)):
#         input_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
#         pred_val = model(input_tensor).item() # 1ステップ先の予測値（正規化済み）
#         predictions_scaled.append(pred_val)
        
#         # 次のステップのために、予測した気温をウィンドウの末尾に追加して1つ古いものを捨てる
#         # ※ ここでは簡単のため、説明変数のうち湿度と気圧は直前の実測値をそのまま流用またはスライドさせる簡易実装
#         next_step = current_window[-1].copy()
#         next_step[0] = pred_val # 気温の予測値を代入
#         current_window = np.vstack([current_window[1:], next_step])

#     # 予測値を元のスケール（実際の気温）に戻す
#     pred_temps = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# # テスト期間に対応する実測値と日付の抽出
# test_start_idx = start_idx + INPUT_WINDOW
# test_dates = df['date'].values[test_start_idx : test_start_idx + len(X_test)]
# actual_temps = df['temperature'].values[test_start_idx : test_start_idx + len(X_test)]


# ---  テストデータ全体の「1ステップ先予測」を一括実行（自己回帰なし） ---
model.eval()
with torch.no_grad():
    # X_test（テストデータの各時点における過去30日分の実測値ウィンドウ）をそのままモデルへ投入
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds_scaled = model(X_test_tensor).cpu().numpy().squeeze(-1)
    
    # 予測値と実際の正解値を元のスケール（実際の気温）に戻す
    pred_temps = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    actual_y_scaled = y_all[train_size:]
    actual_temps = scaler_y.inverse_transform(actual_y_scaled.reshape(-1, 1)).flatten()

# テスト期間に対応する日付の抽出
test_start_idx = train_size + INPUT_WINDOW
test_dates = df['date'].values[test_start_idx : test_start_idx + len(X_test)]


# --- 7. グラフの描画（学習曲線 ＆ 時系列予測比較） ---
plt.figure(figsize=(14, 10))

# 1. 学習曲線のプロット
plt.subplot(2, 1, 1)
plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss (MSE)', color='blue')
plt.plot(range(1, EPOCHS + 1), test_loss_history, label='Test Loss (MSE)', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves (Train vs Test Loss)')
plt.legend()
plt.grid(True)

# 2. テスト区間の実測値 vs 自己回帰予測値のプロット
plt.subplot(2, 1, 2)
plt.plot(test_dates, actual_temps, label='Actual Temperature', color='blue', alpha=0.7)
plt.plot(test_dates, pred_temps, label='Predicted Temperature (Autoregressive)', color='red', linestyle='--', alpha=0.9)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Test Dataset: Actual vs Autoregressive Predicted Temperature')
plt.xticks(test_dates[::max(1, len(test_dates)//10)], rotation=45) # 日付ラベルが混み合わないように間引く
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()