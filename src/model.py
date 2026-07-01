import torch
import torch.nn as nn

#定義模型結構
class DiabetesModel(nn.Module):

    def __init__(self, hidden1=16, hidden2=8, dropout=0.0):

        super().__init__()

        self.Layer1 = nn.Linear(8, hidden1)

        self.Layer2 = nn.Linear(hidden1, hidden2)

        self.Layer3 = nn.Linear(hidden2, 1)

        self.relu = nn.ReLU()
    # dropout 越高，模型越不容易過擬合；太高則可能學不動
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()

#定義向前傳播
    def forward(self, x):

        x = self.Layer1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.Layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.Layer3(x)
        x = self.sigmoid(x)

        return x


# 相容舊名稱，避免其他檔案還在引用舊寫法時壞掉
Diabetesmodel = DiabetesModel