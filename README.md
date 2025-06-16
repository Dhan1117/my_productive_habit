# my_productive_habit

나만의 디지털 습관 분석기
🗂 프로젝트 맵
text
📌 핵심 포인트
어떤 프로젝트인가?

내가 컴퓨터로 하는 모든 활동을 기록해, 내가 집중하고 있는지 딴짓하고 있는지 자동으로 예측합니다.

왜 쓸까?

나만의 디지털 습관을 시각화하고, 생산성을 높이는 데 도움이 됩니다.

어떻게 돌아가나?

Python으로 키보드, 마우스 입력을 감지 → 1분마다 데이터 저장 → 라벨링 → KNN으로 상태 예측

🛠️ 설치 & 실행
필요한 것

Python 3.8 이상

라이브러리: pynput, pandas, scikit-learn, numpy, matplotlib, seaborn, plyer

설치

bash
pip install pynput pandas scikit-learn numpy matplotlib seaborn plyer
실행

데이터 수집: python src/data_collector.py

상태 예측: python src/productivity_predictor.py

📂 폴더 구조
text
my_productive_habit/
├── data/
│   ├── raw/          # 원시 데이터
│   └── labeled/      # 라벨링된 데이터
├── src/
│   ├── data_collector.py
│   └── productivity_predictor.py
├── output/           # 분석 결과
└── README.md
💡 활용 아이디어
생산성 리포트 만들기

디지털 루틴 최적화

머신러닝 실습 데이터로 활용

🤝 기여 & 피드백
이슈 등록: 버그, 기능 제안, 문서 개선

풀 리퀘스트: 코드 수정, 기능 추가

데이터 공유: 나의 활동 데이터와 라벨을 공유해 데이터셋 확장

📝 라이선스
MIT License

나만의 디지털 습관을 분석해, 더 똑똑하게 일하고 공부하세요!
