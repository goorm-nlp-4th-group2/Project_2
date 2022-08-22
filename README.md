# Goorm NLP Project 2 - KLUE MRC (Question Answering)
Goorm 자연어 처리 전문가 양성과정 4기 두 번째 프로젝트
1. 기간 : 2022.07.20~2022.08.03
2. 주제 : KLUE 기계독해 데이터를 활용한 Extractive QA 모델 개발
3. 목표
    1) 입력 문장 길이의 한계 극복
    2) 후처리의 전반적인 과정 및 방법 이해
4. 성과
    1) Cosine similarity 기반 context 내 sentence resampling 방법 제시 및 적용
    2) 조사, 부사 등 불필요한 어말어미 제거
    
5. 환경 : Google Colab Pro+
6. 주요 라이브러리 : transformers, datasets, pandas, torch, re, scikit-learn
7. 구성원 및 역할
    * 박정민 (팀장)
        * 팀 프로젝트 관리 감독 및 총괄
        * 데이터 기반 입력 문장 길히 제한 극복 방법 탐색
        * 팀원 간 Baseline으로 사용할 사전학습 기반 QA 모델 학습 code 작성
    * 이예인
        * Hyper parameter tuning 및 사전학습 모델 탐색
        * QA Task와 관련된 배경지식 및 정보 수집
    * 이용호
        * 사전학습 기반의 QA Baseline code 탐색
        * 학습용 추가 데이터 조사
    * 임영서
        * 데이터 탐색, XAI 기법 탐색
        * 멘토님께서 제공해주신 Baseline 분석 및 적용
        * 모델 기반의 입력 문장 길이 제한 극복 방법 탐색 (사전학습 모델의 embedding 재조정)
    * 정재빈
        * Metric(Levenshtein distance) 정보 조사
        * 모델 기반의 입력 문장 길이 제한 극복 방법 탐색 (512 초과의 길이로 학습된 사전학습 모델)
8. 핵심 아이디어 및 기술
    * WandB Sweep을 활용한 Hyperparameter 탐색 자동화
    * Cosine similarity를 활용한 context sentence resampling
