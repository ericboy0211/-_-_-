# 元智大學 114-2 電腦視覺與影像處理概論期末專題
## 智慧校園應用：基於 YOLO 深度學習之健身房擁擠程度即時監測系統
**Real-Time Campus Gymnasium Crowd Monitoring System Based on YOLO Object Detection**

* **學生姓名**：吳帛諺
* **學生學號**：(請填入你的學號)
* **指導教授**：林柏江 助理教授
* **應用環境**：元智大學體育館健身房（健康休閒中心）

---

#專案簡介
本專案為「電腦視覺與影像處理概論 (EEB215A)」之期末專題。針對元智大學體育館健身房尖峰時段人潮擁擠、器材排隊時間長的問題，本系統導入一階段物件偵測演算法 **Ultralytics YOLO**，實現即時的人數統計（People Counting）與環境擁擠度分級。

為符合特定校園場景需求，本專案**自主實地收集並人工標註**了元智健身房專屬影像資料集，透過微調（Fine-tuning）提升模型在複雜重訓姿勢與器材遮擋下的辨識精準度。另外，系統中引入了**時序平滑演算法（Temporal Smoothing）**，有效消除連續影格中因短暫遮擋造成的人數閃爍與突波。

---

#倉庫目錄結構
```text
├── dataset/                  # 自製元智健身房影像資料集 (或存放資料集下載連結)
│   ├── images/               # 抽樣之訓練與測試圖片 (.jpg)
│   └── labels/               # 符合 YOLO 格式之人工標註檔 (.txt)
├── weights/
│   └── yolov8n.pt            # 預訓練權重模型
├── s1130449_final.py         # 專題核心執行主程式碼
├── requirements.txt          # 相依套件清單
└── README.md                 # 本說明文件

git clone [https://github.com/您的帳號/您的倉庫名稱.git](https://github.com/您的帳號/您的倉庫名稱.git)
cd 您的倉庫名稱

conda create -n yzu_gym python=3.10 -y
conda activate yzu_gym

pip install ultralytics opencv-python
