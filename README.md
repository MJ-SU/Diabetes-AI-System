## Diabetes AI System

這是一個以 PyTorch 建立的糖尿病風險預測專案，提供：

- 訓練與評估模型
- 固定的 `scaler.pkl` 前處理流程
- 命令列預測與 SHAP 解釋
- Streamlit 前端介面

### 專案結構

# Diabetes AI System

這是一個以 PyTorch 建立的糖尿病風險預測專案，整合了資料前處理、模型訓練、固定 scaler、風險預測、SHAP 可解釋性分析，以及 Streamlit 前端介面。

這份專案的設計重點是讓使用者能夠直接輸入病人資料，快速得到：

- 糖尿病風險分數
- 低 / 中 / 高風險分層
- SHAP 風險解釋圖
- 可複製的文字報告

## 專案特色

- 使用 PyTorch 建立二分類神經網路模型
- 訓練時固定儲存 `scaler.pkl`，確保預測與解釋流程一致
- 提供命令列版預測與 SHAP 分析
- 提供 Streamlit 前端，方便展示與互動
- 前端內建輸入參考範圍與即時警示
- 產出文字報告與圖檔，利於展示與分享

## 技術架構

### 資料流程

1. 讀取 `data/diabetes.csv`
2. 清理資料與補缺值
3. 切分訓練集與測試集
4. 使用訓練集 fit `StandardScaler`
5. 將 scaler 儲存為 `models/scaler.pkl`
6. 訓練 PyTorch 模型並儲存為 `models/diabetes_model.pth`
7. 預測與 SHAP 都載入同一份 scaler

### 模型輸出

- `models/diabetes_model.pth`：訓練後的模型權重
- `models/scaler.pkl`：固定前處理 scaler
- `outputs/predict_report.txt`：預測報告
- `outputs/predict_shap_waterfall.png`：SHAP 解釋圖

## 專案結構

- `app.py`：Streamlit 前端入口
- `src/train.py`：模型訓練與權重輸出
- `src/predict.py`：命令列預測、風險分層、SHAP 解釋、報告輸出
- `src/explain.py`：獨立 SHAP 分析腳本
- `src/preprocess.py`：資料前處理、scaler 儲存與載入
- `src/model.py`：PyTorch 模型定義
- `models/`：模型權重與 scaler
- `outputs/`：報告與圖檔輸出
- `data/diabetes.csv`：原始資料集

## 環境需求

- Python 3.13 以上
- `pip`
- 建議使用虛擬環境

## 安裝方式

1. 建立虛擬環境並啟動。

2. 安裝依賴套件。

   ```bash
   pip install -r requirements.txt
   ```

3. 確認 `models/diabetes_model.pth` 與 `models/scaler.pkl` 已存在。

## 啟動方式

### 1. 啟動 Streamlit 前端

```bash
streamlit run app.py
```

預設會在本機開啟，通常可透過 `Local URL` 在瀏覽器中存取。

如果要讓同一區網內其他裝置也能存取，可改成：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

這時除了本機外，也可以透過 `Network URL` 或區網 IP 開啟。

停止執行時，回到啟動 Streamlit 的終端機按 `Ctrl+C`。

### 2. 命令列預測模式

```bash
python src/predict.py
```

### 3. 重新訓練模型

```bash
python src/train.py
```

訓練完成後會重新輸出模型與 scaler。

### 4. 只做 SHAP 分析

```bash
python src/explain.py
```

## 前端功能說明

### 輸入區

前端提供 8 項病人特徵輸入：

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

每個欄位都會顯示常見參考範圍，其中 `DiabetesPedigreeFunction` 已設定為 0.1 步進，方便更細緻輸入。

### 輸出區

按下預測後，前端會顯示：

- 風險分數
- 風險分層
- 建議說明
- SHAP 解釋圖
- 文字報告全文
- 報告下載按鈕

## SHAP 解釋說明

本專案使用 SHAP 來說明模型對單一病人預測結果的影響：

- `E[f(x)]` 代表背景資料的平均預測基準值
- `f(x)` 代表該筆病人資料的模型輸出
- 正值表示特徵把風險往上推
- 負值表示特徵把風險往下拉

這種解釋方式是機器學習領域常見的可解釋性方法，適合用於醫療風險初篩或模型展示。

## 推上 GitHub 的建議流程

### 1. 確認要提交的內容

建議保留以下檔案與資料夾：

- `app.py`
- `src/`
- `models/diabetes_model.pth`
- `models/scaler.pkl`
- `data/diabetes.csv`
- `requirements.txt`
- `README.md`

建議不要提交：

- `.venv/`
- `__pycache__/`
- 臨時執行產物或不必要的大型檔案

### 2. Git 推送步驟

如果你的環境已安裝 Git，可以依序執行：

```bash
git init
git add .
git commit -m "Initial commit: diabetes ai system"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

如果你已經有遠端倉庫，則只要更新 commit 再 push 即可。

### 3. GitHub 上的展示重點

建議你在 GitHub README 或專案說明中突出以下內容：

- 你使用 PyTorch 完成模型建構
- 你有固定 scaler，確保訓練與預測一致
- 你有做 SHAP 可解釋性
- 你有做 Streamlit 互動式前端
- 你有輸出文字報告與圖表，適合實際展示

## 常見問題

### 為什麼 Streamlit 會顯示 Local URL 和 Network URL？

這代表它已在本機啟動。Local URL 是本機瀏覽器用，Network URL 則可供區網內其他設備存取。

### 為什麼按下預測後還會產生報告檔？

因為系統除了顯示畫面，也會把結果輸出成 `outputs/predict_report.txt`，方便留存、複製與分享。

### 為什麼要保留 scaler.pkl？

因為訓練與預測必須使用同一個標準化器，否則輸入尺度不一致，模型結果與 SHAP 解釋都會失真。

## 注意事項

- 這是醫療風險初篩工具，不是診斷工具。
- 若輸入值超出常見參考範圍，前端會顯示提醒，但仍允許送出。
- 推上 GitHub 前，請確認模型檔與 scaler 檔都已提交。

## 建議下一步

如果你要把這份專案作為作品集或面試展示，下一步最值得補的通常是：

1. 加上專案截圖
2. 補一段模型評估結果與指標說明
3. 如果要正式部署，可加上雲端部署說明或 API 版本
