import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pynput import mouse, keyboard
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 지정
plt.rcParams['axes.unicode_minus'] = False  

# --- 전역 변수 및 설정 ---
DATA_DIR = 'data/raw'
LABELED_DIR = 'data/labeled'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LABELED_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, 'activity_log.csv')
LABELED_PATH = os.path.join(LABELED_DIR, 'labeled_activity.csv')

# 샘플 데이터 생성 (없는 경우)
if not os.path.exists(LABELED_PATH):
    sample_data = {
        'key_count': [15,45,25,60,10,80,30,20,70,35,12,55,28,65,8,75,32,18,50,40,22,85,37,14,48,29,62,9,72,42],
        'click_count': [8,15,25,10,5,20,35,10,15,30,6,18,28,12,4,22,32,9,16,38,11,25,33,7,14,29,19,3,21,36],
        'mouse_distance': [300,500,1500,800,200,1200,1800,400,600,2000,250,700,1700,900,150,1100,1900,350,550,2100,450,1300,1950,280,480,1600,750,180,1000,2200],
        'label': [0,1,2,1,0,1,2,0,1,2,0,1,2,1,0,1,2,0,1,2,0,1,2,0,1,2,1,0,1,2]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(LABELED_PATH, index=False)

# 전역 변수
key_count = 0
click_count = 0
mouse_distance = 0.0
last_mouse_pos = None
stop_event = threading.Event()

# 상태 매핑
label_map = {
    0: {'name': '딴짓 중', 'emoji': '🤔', 'color': '#FF6B6B'},
    1: {'name': '집중 중', 'emoji': '🔥', 'color': '#4ECDC4'},
    2: {'name': '자료조사 중', 'emoji': '🔎', 'color': '#FFD166'}
}

# --- 데이터 수집 로직 ---
def on_press(key):
    global key_count
    key_count += 1

def on_click(x, y, button, pressed):
    global click_count
    if pressed:
        click_count += 1

def on_move(x, y):
    global mouse_distance, last_mouse_pos
    if last_mouse_pos:
        distance = math.sqrt((x - last_mouse_pos[0])**2 + (y - last_mouse_pos[1])**2)
        mouse_distance += distance
    last_mouse_pos = (x, y)

def save_data_periodically(interval=60):
    global key_count, click_count, mouse_distance
    
    # 데이터 폴더 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # CSV 헤더 작성 (파일 없을 때만)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'key_count', 'click_count', 'mouse_distance'])
    
    # 리스너 시작
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    keyboard_listener.start()
    mouse_listener.start()
    
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
            
        # 데이터 저장
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        data_row = [timestamp, key_count, click_count, int(mouse_distance)]
        
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
            
        # UI 업데이트
        if status_label:
            status_text = (f"최근 수집 데이터:\n"
                          f"키보드: {key_count}회, "
                          f"클릭: {click_count}회\n"
                          f"이동: {int(mouse_distance)}px")
            status_label.config(text=status_text)
        
        # 변수 초기화
        key_count = 0
        click_count = 0
        mouse_distance = 0.0
        
    # 리스너 정지
    keyboard_listener.stop()
    mouse_listener.stop()

# --- KNN 모델 및 예측 ---
scaler = None
knn = None

def train_knn_model():
    global scaler, knn
    try:
        df = pd.read_csv(LABELED_PATH)
        X_train = df[['key_count', 'click_count', 'mouse_distance']]
        y_train = df['label']
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # KNN 모델 훈련
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        return True
    except Exception as e:
        print(f"모델 훈련 오류: {e}")
        return False

def predict_activity():
    global key_count, click_count, mouse_distance
    
    if scaler is None or knn is None:
        return None, 0.0
    
    try:
        # 현재 활동 데이터로 예측
        current_features = np.array([[key_count, click_count, int(mouse_distance)]])
        current_features_scaled = scaler.transform(current_features)
        
        # 예측 수행
        prediction = knn.predict(current_features_scaled)
        probabilities = knn.predict_proba(current_features_scaled)
        
        predicted_label = prediction[0]
        confidence = probabilities[0][predicted_label] * 100
        return predicted_label, confidence
    except Exception as e:
        print(f"예측 오류: {e}")
        return None, 0.0

# --- UI 애플리케이션 ---
class DigitalHabitAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("디지털 습관 분석기")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 상태 변수
        self.is_running = False
        self.data_thread = None
        self.prediction_thread = None
        
        # UI 생성
        self.create_widgets()
        
        # 모델 미리 학습
        if not train_knn_model():
            messagebox.showerror("오류", "KNN 모델 학습에 실패했습니다. 데이터를 확인해주세요.")
        
    def create_widgets(self):
        # 상단 프레임
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 제목
        title_label = ttk.Label(top_frame, text="디지털 습관 분석기", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 제어 버튼
        self.control_button = ttk.Button(
            top_frame, 
            text="시작", 
            width=15,
            command=self.toggle_analysis,
            style="TButton"
        )
        self.control_button.pack(pady=10)
        
        # 상태 표시 프레임
        status_frame = ttk.LabelFrame(self.root, text="실시간 상태")
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 데이터 상태 레이블
        global status_label
        status_label = ttk.Label(
            status_frame, 
            text="대기 중...",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        status_label.pack(pady=10, padx=10, anchor=tk.W)
        
        # 예측 결과 프레임
        prediction_frame = ttk.LabelFrame(self.root, text="활동 예측 결과")
        prediction_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 예측 결과 레이블
        self.prediction_label = tk.Label(
            prediction_frame,
            text="분석을 시작해주세요",
            font=("Arial", 20, "bold"),
            justify=tk.CENTER,
            height=3
        )
        self.prediction_label.pack(fill=tk.X, pady=10, padx=20)
        
        # 신뢰도 표시
        self.confidence_label = ttk.Label(
            prediction_frame,
            text="",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        self.confidence_label.pack(fill=tk.X, pady=5)
        
        # 데이터 시각화 프레임
        viz_frame = ttk.LabelFrame(self.root, text="활동 패턴 시각화")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Matplotlib 그래프
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 초기 그래프 설정
        self.update_visualization()
        
    def toggle_analysis(self):
        if self.is_running:
            # 분석 중지
            self.is_running = False
            self.control_button.config(text="시작")
            stop_event.set()
            
            if self.data_thread:
                self.data_thread.join(timeout=1.0)
            
            if self.prediction_thread:
                self.prediction_thread.join(timeout=1.0)
                
            status_label.config(text="분석이 중지되었습니다")
            self.prediction_label.config(
                text="분석이 중지되었습니다",
                bg="SystemButtonFace"
            )
            self.confidence_label.config(text="")
        else:
            # 분석 시작
            self.is_running = True
            self.control_button.config(text="중지")
            stop_event.clear()
            
            # 데이터 수집 스레드 시작
            self.data_thread = threading.Thread(
                target=save_data_periodically, 
                args=(60,),
                daemon=True
            )
            self.data_thread.start()
            
            # 예측 스레드 시작
            self.prediction_thread = threading.Thread(
                target=self.run_prediction_loop,
                daemon=True
            )
            self.prediction_thread.start()
            
            status_label.config(text="데이터 수집 시작...")
    
    def run_prediction_loop(self):
        while self.is_running and not stop_event.is_set():
            stop_event.wait(10)  # 10초마다 예측
            
            # 활동 예측
            predicted_label, confidence = predict_activity()
            
            if predicted_label is not None:
                # UI 업데이트
                label_info = label_map.get(predicted_label, {
                    'name': '알 수 없음',
                    'emoji': '❓',
                    'color': '#CCCCCC'
                })
                
                self.prediction_label.config(
                    text=f"{label_info['emoji']} {label_info['name']}",
                    bg=label_info['color']
                )
                
                self.confidence_label.config(
                    text=f"신뢰도: {confidence:.2f}%"
                )
            
            # 시각화 업데이트
            self.update_visualization()
    
    def update_visualization(self):
        try:
            # 기존 그래프 클리어
            self.ax.clear()
            
            # 샘플 데이터 로드
            df = pd.read_csv(LABELED_PATH)
            
            # 활동 유형별 평균 계산
            activity_types = {
                0: '딴짓',
                1: '집중',
                2: '자료조사'
            }
            
            avg_data = []
            for label, name in activity_types.items():
                subset = df[df['label'] == label]
                if not subset.empty:
                    avg_values = subset[['key_count', 'click_count', 'mouse_distance']].mean()
                    avg_data.append({
                        'label': name,
                        '키보드': avg_values['key_count'],
                        '클릭': avg_values['click_count'],
                        '이동거리': avg_values['mouse_distance'] / 1000  # 천 단위로 정규화
                    })
            
            # DataFrame으로 변환
            viz_df = pd.DataFrame(avg_data)
            
            # 그래프 그리기
            if not viz_df.empty:
                viz_df.set_index('label').plot(
                    kind='bar', 
                    ax=self.ax,
                    color=['#FF6B6B', '#4ECDC4', '#FFD166']
                )
                self.ax.set_title('Average Activity Patterns by Type')
                self.ax.set_ylabel('Activity Level')
                self.ax.set_xlabel('Activity Type')
                self.ax.legend(title='Metrics')
                self.ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 캔버스 업데이트
            self.canvas.draw()
            
        except Exception as e:
            print(f"시각화 업데이트 오류: {e}")

    def on_closing(self):
        stop_event.set()
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
        self.root.destroy()

# --- 메인 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitalHabitAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 스타일 설정
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)
    style.configure("TLabelframe.Label", font=("Arial", 11, "bold"))
    
    root.mainloop()
