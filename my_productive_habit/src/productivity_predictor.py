import time
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pynput import mouse, keyboard
import threading

# --- 1. 모델 훈련 ---
print("라벨링된 데이터로 모델을 훈련합니다...")

# 라벨링된 데이터 로드
try:
    df = pd.read_csv('labeled_activity.csv')
except FileNotFoundError:
    print("오류: 'labeled_activity.csv' 파일을 찾을 수 없습니다.")
    print("2단계 데이터 라벨링 작업을 먼저 완료해주세요.")
    exit()

# 특성(X)과 라벨(y) 분리
X_train = df[['key_count', 'click_count', 'mouse_distance']]
y_train = df['label']

# 데이터 스케일링 (KNN에서 매우 중요) [3]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# KNN 모델 생성 및 훈련 [4]
# k=5는 일반적인 시작 값이며, 데이터에 따라 조절할 수 있습니다.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 라벨 맵 정의
label_map = {0: '딴짓 중... 🤔', 1: '집중 업무 중! 🔥', 2: '자료 조사 중... 🔎'}

print("모델 훈련 완료. 실시간 예측을 시작합니다.")

# --- 2. 실시간 예측 ---
# 데이터 수집을 위한 전역 변수 (1단계와 동일)
key_count = 0
click_count = 0
mouse_distance = 0.0
last_mouse_pos = None

def on_press(key):
    global key_count
    key_count += 1

def on_click(x, y, button, pressed):
    global click_count
    if pressed:
        click_count += 1

def on_move(x, y):
    global mouse_distance, last_mouse_pos
    if last_mouse_pos is not None:
        distance = math.sqrt((x - last_mouse_pos[0])**2 + (y - last_mouse_pos[1])**2)
        mouse_distance += distance
    last_mouse_pos = (x, y)

# 리스너 시작
keyboard_listener = keyboard.Listener(on_press=on_press)
mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
keyboard_listener.start()
mouse_listener.start()

try:
    while True:
        # 60초 동안 데이터 수집
        time.sleep(60)
        
        # 현재 수집된 데이터로 특성 벡터 생성
        current_features = np.array([[key_count, click_count, int(mouse_distance)]])
        
        # 훈련 시 사용했던 scaler로 동일하게 스케일링
        current_features_scaled = scaler.transform(current_features)
        
        # 모델로 현재 상태 예측 [4]
        prediction = knn.predict(current_features_scaled)
        predicted_label = prediction[0]
        
        # 예측 확률 확인 (어떤 클래스에 얼마나 확신하는지) [5]
        probabilities = knn.predict_proba(current_features_scaled)
        confidence = probabilities[0][predicted_label] * 100
        
        # 결과 출력
        timestamp = time.strftime('%H:%M:%S', time.localtime())
        status_text = label_map.get(predicted_label, "알 수 없는 상태")
        
        print(f"[{timestamp}] 현재 상태 예측: {status_text} (신뢰도: {confidence:.2f}%)")
        print(f"    (입력: {key_count}, 클릭: {click_count}, 이동: {int(mouse_distance)})")
        
        # 다음 주기를 위해 카운터 초기화
        key_count = 0
        click_count = 0
        mouse_distance = 0.0

except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")
    keyboard_listener.stop()
    mouse_listener.stop()
