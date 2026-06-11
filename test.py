import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(page_title="약물 농도 AI 모델링 시뮬레이터", layout="wide")

st.title("💊 AI 융합 프로젝트: 약물 농도 변화 모델링 및 시뮬레이션 웹 앱")
st.write("고등학교 수학(지수함수)·화학(반응속도론)·생명과학 융합 수업을 위한 탐구 도구입니다.")

# -----------------------------------------------------------------------------
# [공통 데이터 생성] 2차시 실습용 가상 환자 데이터셋
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 60) # 0시간부터 12시간까지 60개 시점
C_0_true = 25.0
k_true = 0.3466  # 타이레놀 기본 반감기 2시간 기준 (ln(2)/2)
noise = np.random.normal(0, 0.8, len(time_data))
conc_data = C_0_true * np.exp(-k_true * time_data) + noise
conc_data = np.maximum(conc_data, 0.1) # 음수 방지

df_sample = pd.DataFrame({"Time_hours": time_data, "Concentration": conc_data})

# -----------------------------------------------------------------------------
# 사이드바 설정 (3차시: 변인 통제 및 타이레놀 분석용)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 3차시: 시뮬레이션 변수 제어")
st.sidebar.markdown("### 🧪 복용 조건 및 환자 특성 변경")
input_c0 = st.sidebar.slider("최초 투여 직후 농도 ($C_0$, $\mu g/mL$)", 10.0, 50.0, 25.0)
input_half_life = st.sidebar.slider("환자의 약물 반감기 ($t_{1/2}$, 시간)", 1.0, 6.0, 2.0, 
                                    help="정상 성인은 약 2시간이며, 간이나 신장 기능이 저하된 환자는 늘어납니다.")

# 입력받은 반감기로 배설 속도 상수 k 계산 (k = ln(2) / t_1/2)
calculated_k = np.log(2) / input_half_life

# -----------------------------------------------------------------------------
# 메인 화면 - 2차시 및 3차시 탭(Tab) 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "📐 2차시: 데이터 분포 확인 및 경향성 곡선 탐구", 
    "🎯 3차시: 타이레놀 효과 및 투약 시간 예측"
])

# -----------------------------------------------------------------------------
# 2차시 탭: 그래프 상의 분포 구현 및 지수함수 회귀분석 (수동 피팅 및 탐구형 수정)
# -----------------------------------------------------------------------------
with tab1:
    st.header("2차시: 약물 농도 감소는 어떤 수학적 모델로 설명할 수 있을까?")
    st.markdown("""
    **주요 활동**: 제시된 시간에 따른 약물 농도 데이터를 산점도로 시각화하여 경향성을 관찰합니다. 
    이후 점들의 분포를 가장 잘 대변하는 곡선을 직접 그려보고, 이 곡선이 어떤 수학적 함수 식을 가질지 탐구해 봅시다.
    """)
    
    # [단계 1] 데이터 표 격리
    with st.expander("📋 [단계 1] 수집된 약물 농도 데이터 전체 보기 (숫자 데이터 확인)"):
        st.write("MBL 센서나 스마트 기기로 수집된 가상의 시간별 농도 데이터셋(총 60행)입니다.")
        st.dataframe(df_sample, use_container_width=True)
        
    st.markdown("---")
    
    # [단계 2 & 3 통합 시각화 공간]
    st.subheader("📊 [단계 2] 데이터 분포 시각화 및 곡선 그려보기")
    st.write("시간이 흐름에 따라 혈중 약물 농도가 어떻게 감소하는지 빨간색 점(산점도)을 관찰하고, 아래 슬라이더를 움직여 점들의 흐름을 관통하는 나만의 곡선을 완성해 보세요.")

    # 학생들이 직접 그래프를 그리기 위한 수동 매개변수 슬라이더 (AI 자동 연산 제거)
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        manual_a = col_sl1.slider("곡선의 시작 높이 조절 (파라미터 $a$)", 10.0, 40.0, 15.0, step=0.5)
    with col_sl2:
        manual_b = col_sl2.slider("곡선의 감소 속도 조절 (파라미터 $b$)", 0.01, 1.00, 0.10, step=0.01)

    # 그래프 시각화 (산점도 위에 학생들이 실시간으로 조절하는 곡선이 중첩됨)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    # 1. 원본 산점도 분포
    ax.scatter(df_sample["Time_hours"], df_sample["Concentration"], color="crimson", alpha=0.7, label="실제 측정 데이터 (오차 포함)")
    
    # 2. 학생들이 슬라이더로 조작하는 동적 탐구 곡선
    time_trend = np.linspace(0, 12, 200)
    user_curve = manual_a * np.exp(-manual_b * time_trend)
    ax.plot(time_trend, user_curve, color="darkblue", linewidth=2.5, 
             label=f"내가 그린 경향성 곡선: $y = {manual_a:.1f} \\cdot e^{{-{manual_b:.2f}x}}$")
    
    ax.set_xlabel("시간 (Time, hours)")
    ax.set_
