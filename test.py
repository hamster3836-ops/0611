import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("💊 타이레놀(아세트아미노펜) 체내 대사 AI 시뮬레이터")

# 1. 사이드바 - 환자군 및 투여 조건 설정
st.sidebar.header("환자 조건 설정")
patient = st.sidebar.selectbox("환자 유형 선택", ["정상 성인 (반감기 2시간)", "간/신장 저하 환자 (반감기 5시간)"])

# 약학정보원 기반 배설 속도 상수(k) 결정
if "정상" in patient:
    true_k = np.log(2) / 2.0  # k = 0.3466
else:
    true_k = np.log(2) / 5.0  # k = 0.1386

# 2. 100개의 시뮬레이션 데이터셋 생성 (0~12시간 관찰)
np.random.seed(42)
time = np.linspace(0, 12, 100)
C_0 = 20.0  # 초기 최고 혈중 농도 가정 (ug/mL)

# 오차가 포함된 실제 측정값 모사
noise = np.random.normal(0, 0.5, len(time))
concentration = C_0 * np.exp(-true_k * time) + noise
concentration = np.maximum(concentration, 0.1)  # 음수 방지

df = pd.DataFrame({"Time_hours": time, "Concentration": concentration})

# 3. 로그 변환 수행 (수학적 모델링)
df["ln_Concentration"] = np.log(df["Concentration"])

# 4. AI 선형 회귀 학습 버튼
if st.button("🤖 AI 모델 학습 및 반감기 추정 시작"):
    X = df[["Time_hours"]]
    y = df["ln_Concentration"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    pred_k = -model.coef_[0]
    pred_half_life = np.log(2) / pred_k
    
    st.success(f"AI가 계산한 배설 속도 상수(k): **{pred_k:.4f}**")
    st.metric(label="AI 예측 최종 반감기", value=f"{pred_half_life:.2f} 시간")
    
    # 약학정보원 가이드라인 연계 피드백
    if pred_half_life > 4.0:
        st.error("⚠️ 경고: 약물 대사 속도가 매우 느립니다. 타이레놀의 일반적 복용 간격(4~6시간)보다 인출 주기를 늘려야 간독성 위험(NAPQI 축적)을 방지할 수 있습니다.")
    else:
        st.info("✅ 정상 대사 범주입니다. 하루 최대 허용량(4g) 이내에서 4~6시간 간격 복용이 안전합니다.")
