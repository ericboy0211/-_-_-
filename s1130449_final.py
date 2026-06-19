import cv2
from ultralytics import YOLO

# 1. 載入 YOLO 模型 (預設會自動下載權重檔，yolov8n.pt 或 yolo11n.pt 皆可)
# Nano (n) 版本速度最快，非常適合筆電即時偵測
model = YOLO("yolov8n.pt") 

# 2. 讀取測試影片或開啟攝影機
# 若要使用筆電內建攝影機，請改為 cv2.VideoCapture(0)
video_path = 0 
cap = cv2.VideoCapture(video_path)

# 定義元智健身房的擁擠度門檻（可依現場實際大小調整）
THRESHOLD_SPARSE = 1   # 3人以下算舒適
THRESHOLD_CROWDED = 7  # 7人以上算擁擠

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("影片播放結束或無法讀取攝影機畫面。")
        break
        
    # 調整畫面大小以符合 YOLO 預設輸入或優化顯示速度
    frame = cv2.resize(frame, (640, 480))
    
    # 3. 使用 YOLO 進行物件偵測
    # classes=0 代表只偵測「人 (person)」，stream=True 可以提高影片串流處理效率
    results = model(frame, classes=[0], stream=True)
    
    people_count = 0
    
    for r in results:
        boxes = r.boxes
        people_count = len(boxes) # 本影格偵測到的人數
        
        # 遍歷每個偵測到的人體並繪製邊界框
        for box in boxes:
            # 取得邊界框座標 (左上角 x1, y1, 右下角 x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 取得信心度 (Confidence score)
            conf = float(box.conf[0])
            
            # 繪製綠色偵測框與標籤 (只顯示信心度高於 0.4 的物件)
            if conf > 0.4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    # 4. 根據當前人數判斷健身房擁擠程度
    if people_count <= THRESHOLD_SPARSE:
        status = "Comfortable (Sparse)"
        color = (0, 255, 0)       # 舒適綠色
    elif people_count < THRESHOLD_CROWDED:
        status = "Moderate"
        color = (0, 255, 255)     # 普通黃色
    else:
        status = "Crowded!"
        color = (0, 0, 255)       # 擁擠紅色
        
    # 5. 在畫面上即時顯示統計數據
    # 建立一個半透明的背景資訊欄，讓文字更清晰
    cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1) 
    
    cv2.putText(frame, f"YZU Gym Monitor", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"People Count: {people_count}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Status: {status}", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 顯示結果視窗
    cv2.imshow("YZU Gym Crowd Detection (YOLO)", frame)
    
    # 按下 'q' 鍵可提前退出程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()