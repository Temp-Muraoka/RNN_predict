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
OUTPUT_WINDOW = 10   # 未来予測対象日数
HIDDEN_SIZE = 6    # 隠れ層の次元数
NUM_LAYERS = 1      # RNNレイヤー数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# --- 前処理とスケーリング ---
# --- 3. 最新10個を予測対象（未来データ）として完全に切り離す ---
future_df = df.tail(OUTPUT_WINDOW).copy()      # 最後の10日分（学習には絶対に使わない）
df = df.iloc[:-OUTPUT_WINDOW].copy()  # 残りのデータで学習・テストを行う


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
# y_all = y_all.squeeze(-1) # 形状調整 (samples,)

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

# --- 4. Seq2Seqモデルの定義 ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        _, hidden = self.lstm(x)
        return hidden # (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, output_window, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_window = output_window
        
    def forward(self, encoder_hidden, target_tensor=None):
        # target_tensor: 訓練時のTeacher Forcing用 (batch, output_window, 1)
        batch_size = encoder_hidden[0].size(1)
        device = encoder_hidden[0].device
        
        # Decoderの最初の入力としてゼロ（または直前の値）を使用
        decoder_input = torch.zeros(batch_size, 1, 1).to(device)
        hidden = encoder_hidden
        
        outputs = []
        for t in range(self.output_window):
            out, hidden = self.lstm(decoder_input, hidden)
            pred = self.fc(out) # (batch, 1, 1)
            outputs.append(pred)
            
            # 訓練時は正解データを使う（Teacher Forcing）、推論時は自分の予測を次の入力にする
            if target_tensor is not None and np.random.rand() < 0.5:
                decoder_input = target_tensor[:, t:t+1, :]
            else:
                decoder_input = pred
                
        outputs = torch.cat(outputs, dim=1) # (batch, output_window, 1)
        return outputs.squeeze(-1)

class WeatherSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(WeatherSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, target=None):
        hidden = self.encoder(src)
        outputs = self.decoder(hidden, target)
        return outputs

# --- 6. 学習の実行 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

encoder = Encoder(input_size=len(feature_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
decoder = Decoder(output_size=1, hidden_size=HIDDEN_SIZE, output_window=OUTPUT_WINDOW, num_layers=NUM_LAYERS)
model = WeatherSeq2Seq(encoder, decoder).to(device)

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

# --- 6. テストデータに対する予測とプロット用データの準備 ---
model.eval()
with torch.no_grad():
    # テストデータの最初から最後まで、OUTPUT_WINDOWおき（あるいは連続）にSeq2Seqで予測を行う
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds_scaled = model(X_test_tensor, target=None).cpu().numpy()
    
    # ここでは代表例として、テストデータの最初のサンプルの「未来10日間の予測」を取り出して可視化
    # またはテスト区間全体のつながりを確認できるようにプロット
    sample_idx = 0
    pred_seq = scaler_y.inverse_transform(preds_scaled[sample_idx].reshape(-1, 1)).flatten()
    actual_seq = scaler_y.inverse_transform(y_test[sample_idx].reshape(-1, 1)).flatten()

# 日付の取得（サンプル区間）
test_start_idx = train_size + INPUT_WINDOW
sample_dates = df['date'].values[test_start_idx + sample_idx : test_start_idx + sample_idx + OUTPUT_WINDOW]

# --- 7. グラフの描画（学習曲線 ＆ Seq2Seq予測結果） ---
plt.figure(figsize=(14, 10))

# 1. 学習曲線
plt.subplot(2, 1, 1)
plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss (MSE)', color='blue')
plt.plot(range(1, EPOCHS + 1), test_loss_history, label='Test Loss (MSE)', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Seq2Seq Learning Curves')
plt.legend()
plt.grid(True)

# 2. Seq2Seqによる未来予測（テストデータの1サンプル＝未来10日間の軌跡）
plt.subplot(2, 1, 2)
plt.plot(sample_dates, actual_seq, marker='o', label='Actual Temperature', color='blue')
plt.plot(sample_dates, pred_seq, marker='x', linestyle='--', label=f'Seq2Seq Predicted ({OUTPUT_WINDOW} Days Ahead)', color='purple')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Seq2Seq: Multi-Step Autoregressive Prediction vs Actual')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. 学習完了後：最後の10個の予測と実際のデータの比較 ---
model.eval()
with torch.no_grad():
    # 最後の10個を予測するため、その直前の「INPUT_WINDOW（30日分）」のデータを取得
    recent_data_window = df.tail(INPUT_WINDOW)[feature_cols].values
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