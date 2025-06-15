import time
import math
import csv
import os
from pynput import mouse, keyboard
import threading

# --- 전역 변수 설정 ---
key_count = 0
click_count = 0
mouse_distance = 0.0
last_mouse_pos = None
stop_collecting = threading.Event()

# CSV 파일 경로
DATA_DIR = 'data/raw'
CSV_PATH = os.path.join(DATA_DIR, 'activity_log.csv')  # 통일된 경로

# --- 리스너 콜백 함수 ---
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

    # 폴더 생성 (추가된 부분)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 파일 초기화 (기존 코드에서 경로 수정)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'key_count', 'click_count', 'mouse_distance'])

    while not stop_collecting.is_set():
        stop_collecting.wait(interval)
        if stop_collecting.is_set():
            break

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        data_row = [timestamp, key_count, click_count, int(mouse_distance)]
        print(f"[{timestamp}] 데이터 저장: {data_row[1:]}")

        # 데이터 저장 경로 통일 (기존 'activity_log.csv' → CSV_PATH로 수정)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

        key_count = 0
        click_count = 0
        mouse_distance = 0.0

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    print("디지털 활동 데이터 수집을 시작합니다. (Ctrl+C로 종료)")
    
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    
    keyboard_listener.start()
    mouse_listener.start()
    
    save_thread = threading.Thread(target=save_data_periodically, args=(60,))
    save_thread.start()

    try:
        save_thread.join()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다...")
        stop_collecting.set()
        keyboard_listener.stop()
        mouse_listener.stop()
        save_thread.join()
        print("데이터 수집이 중단되었습니다.")
