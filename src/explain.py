from pathlib import Path
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

try:
    from src.model import DiabetesModel
    from src.preprocess import prepare_data
except ImportError:
    from model import DiabetesModel
    from preprocess import prepare_data


def load_model(model_path):
    #載入 PyTorch 模型
    state_dict = torch.load(model_path, map_location="cpu")
    hidden1 = state_dict["Layer1.weight"].shape[0]
    hidden2 = state_dict["Layer2.weight"].shape[0]
    model = DiabetesModel(hidden1=hidden1, hidden2=hidden2)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    print("=== SHAP Explainability System ===", flush=True)
    # 1. 載入模型
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "diabetes_model.pth"
    )
    if not model_path.exists():
        print(f"找不到模型檔案: {model_path}", flush=True)
        return

    model = load_model(model_path)
    print("模型載入完成", flush=True)

     # 2. 載入資料
    X_train_tensor, X_test_tensor, _, _, _ = prepare_data(use_saved_scaler=True)

    X_train = X_train_tensor.cpu().numpy()
    X_test = X_test_tensor.cpu().numpy()

    print(f"資料載入完成: train={len(X_train)}, test={len(X_test)}", flush=True)
    # 3. SHAP 設定
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    try:
        # 背景資料（SHAP reference distribution）
        background = X_train[:50]

        # 要解釋的樣本
        samples = X_test[:100]

        def predict_fn(data):
            tensor = torch.tensor(data, dtype=torch.float32)
            with torch.no_grad():
                return model(tensor).cpu().numpy().reshape(-1)

        # 4. 建立 SHAP Explainer
        explainer = shap.Explainer(predict_fn, background)
        shap_values = explainer(samples)

        print("SHAP 計算完成", flush=True)
    # 5. 可視化 (Summary Plot)
        shap_numpy = np.array(shap_values.values)

        plt.figure()
        shap.summary_plot(
            shap_numpy,
            samples,
            feature_names=feature_names,
            show=False
        )

        output_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
        )
        output_path.mkdir(exist_ok=True)

        plt.savefig(output_path / "shap_summary.png", bbox_inches="tight")
        plt.close()

        print(f"SHAP 圖片已輸出至: {output_path}")

    except Exception as exc:
        print(f"SHAP 執行失敗: {exc}", flush=True)
        
if __name__ == "__main__":
    main()