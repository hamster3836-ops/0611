import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 페이지 기본 설정
st.set_page_config(page_title="약물 농도 변화 수학적 모델링", layout="wide")

st.title("💊 AI 융합 프로젝트: 약물 농도 변화의 수학적 모델링 및 시뮬레이션")
st.write("고등학교 화학·수학·AI 융합 수업을 위한 데이터 분석 및 예측 웹 앱입니다.")

# -----------------------------------------------------------------------------
# [공통 데이터 생성] 수업용 가상 환자 데이터셋 (2차시 실습용)
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 60) # 0시간부터 12시간까지 60개 시점
C_0_true = 25.0
k_true = 0.3466  # 반감기 2시간 기준 (ln(2)/2)
noise = np.random.normal(0, 0.8, len(time_data))
conc_data = C_0_true * np.exp(-k_true * time_data) + noise
conc_data = np.maximum(conc_data, 0.1) # 음수 방지

df_sample = pd.DataFrame({"Time_hours": time_data, "Concentration": conc_data})

# -----------------------------------------------------------------------------
# 사이드바 설정 (3차시: 변수 제어 및 타이레놀 분석용)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 시뮬레이션 매개변수 설정")
st.sidebar.markdown("### [3차시 주요 활동] 변인 통제 및 조건 변경")
input_c0 = st.sidebar.slider("최초 투여 직후 농도 ($C_0$, $\mu g/mL$)", 10.0, 50.0, 25.0)
input_half_life = st.sidebar.slider("환자의 약물 반감기 ($t_{1/2}$, 시간)", 1.0, 6.0, 2.0, help="정상 성인은 약 2시간, 대사 저하 환자는 늘어납니다.")

# 입력받은 반감기로 배설 속도 상수 k 계산 (k = ln(2) / t_1/2)
calculated_k = np.log(2) / input_half_life

# -----------------------------------------------------------------------------
# 메인 화면 - 차시별 탭(Tab) 구성
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏃‍♂️ 1차시: 정맥 주사와 몸속 약물 변화", 
    "📐 2차시: 지수함수 모델 회귀분석 (AI)", 
    "🎯 3차시: 타이레놀 효과 및 투약 시간 예측"
])

# -----------------------------------------------------------------------------
# 1차시 탭: 현상 이해 및 문제 설정
# -----------------------------------------------------------------------------
with tab1:
    st.header("1차시: 정맥 주사 후 약물은 몸속에서 어떻게 변화할까?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 주요 탐구 및 개념 소개")
        st.markdown("""
        * **정맥 주사(IV Bolus) vs 경구약 복용**: 경구약은 소화기관을 거쳐 서서히 흡수되지만, 정맥 주사는 투여 즉시 혈중 농도가 최고점($C_0$)을 찍은 후 대사/제거됩니다.
        * **1차 반응(First-order kinetics)**: 체내에서 약물이 제거되는 속도는 현재 몸속에 남아있는 약물의 농도에 비례합니다.
        * **반감기($t_{1/2}$)**: 혈중 약물 농도가 정확히 절반으로 줄어드는 데 걸리는 시간입니다.
        """)
        
    with col2:
        st.subheader("📌 수학적 모델링의 출발 (미분방정식)")
        st.latex(r"\frac{dC}{dt} = -kC")
        st.markdown("이 식을 변수분리법으로 적분하면, 우리가 배운 **지수함수 모델**이 유도됩니다:")
        st.latex(r"C(t) = C_0 \cdot e^{-kt}")
        st.info("💡 2차시 탭으로 이동하여 실제 수집된 데이터 분포를 확인하고 AI에게 회귀 분석을 시켜봅시다.")

# -----------------------------------------------------------------------------
# 2차시 탭: 그래프 상의 분포 구현 및 지수함수 회귀분석
# -----------------------------------------------------------------------------
with tab2:
    st.header("2차시: 약물 농도 감소는 어떤 수학적 모델로 설명할 수 있을까?")
    st.subheader("📊 제시된 시간에 따른 약물 농도 데이터 분포 구현")
    
    col1_t2, col2_t2 = st.columns([1, 2])
    
    with col1_t2:
        st.write("💻 수집된 실제 환자의 약물 농도 데이터셋 (일부)")
        st.dataframe(df_sample.head(10), height=300)
        
    with col2_t2:
        # 2차시 목표: 그래프 상의 분포 구현
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df_sample["Time_hours"], df_sample["Concentration"], color="crimson", alpha=0.7, label="실제 측정 데이터 (오차 포함)")
        ax.set_xlabel("시간 (Time, hours)")
        ax.set_ylabel("농도 (Concentration, ug/mL)")
        ax.set_title("시간에 따른 약물 농도 데이터 분포")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🤖 지수함수 모델로 AI 회귀분석(Regression) 하기")
    st.write("지수함수 곡선 상태로는 선형 회귀 알고리즘이 가중치를 찾기 어렵습니다. 따라서 양변에 **자연로그($\ln$)를 취해 선형화(직선 변환)**합니다.")
    st.latex(r"\ln C(t) = -kt + \ln C_0")
    
    if st.button("📈 AI 회귀분석 실행 (로그 변환 후 경사하강법 학습)"):
        # 수학적 전처리: y 데이터에 로그 취하기
        X = df_sample[["Time_hours"]]
        y_log = np.log(df_sample["Concentration"])
        
        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X, y_log)
        
        # AI가 찾아낸 파라미터 역산
        pred_k = -model.coef_[0]
        pred_c0 = np.exp(model.intercept_)
        pred_half_life = np.log(2) / pred_k
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("🤖 AI가 예측한 초기 농도 ($C_0$)", f"{pred_c0:.2f} \mu g/mL")
        col_res2.metric("📐 AI가 예측한 배설 상수 ($k$)", f"{pred_k:.4f}")
        col_res3.metric("⏳ 계산된 데이터의 반감기 ($t_{1/2}$)", f"{pred_half_life:.2f} 시간")
        
        # 결과 시각화 곡선 그리기
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.scatter(df_sample["Time_hours"], df_sample["Concentration"], color="crimson", alpha=0.5, label="측정 데이터")
        
        # AI가 예측한 함수식 기반 곡선 데이터
        time_trend = np.linspace(0, 12, 200)
        fit_conc = pred_c0 * np.exp(-pred_k * time_trend)
        ax2.plot(time_trend, fit_conc, color="darkblue", linewidth=2.5, label=f"AI 추정 지수 곡선: C(t) = {pred_c0:.1f} * e^(-{pred_k:.2f}t)")
        
        ax2.set_xlabel("시간 (hours)")
        ax2.set_ylabel("농도 (ug/mL)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

# -----------------------------------------------------------------------------
# 3차시 탭: 약학정보원 데이터 기반 타이레놀 복용 시간 예측 및 한계 설명
# -----------------------------------------------------------------------------
with tab3:
    st.header("3차시: 약물 농도 변화 모델을 활용하여 약물 효과 지속 시간 예측하기")
    st.subheader("📋 약학정보원(health.kr) 타이레놀(아세트아미노펜) 가이드라인 연계")
    
    st.markdown("""
    * **타이레놀의 치료 농도 범위(Therapeutic Window)**: 약효를 내는 최소 유효 농도(**MEC**)는 보통 **5 $\mu g/mL$**, 부작용(간독성)을 일으키는 최소 독성 농도(**MTC**)는 **25 $\mu g/mL$** 부근으로 알려져 있습니다.
    * 왼쪽 사이드바에서 **환자의 초기 농도와 반감기를 조절**하며 그래프가 어떻게 변하는지 확인해 보세요!
    """)
    
    # 사이드바 입력값을 기반으로 시뮬레이션 타임라인 생성
    t_sim = np.linspace(0, 24, 300)
    c_sim = input_c0 * np.exp(-calculated_k * t_sim)
    
    # 약효 지속 시간(농도가 5 이상 유지되는 시간) 수학적으로 계산
    # 5 = C_0 * e^(-kt)  ->  ln(5/C_0) = -kt  ->  t = ln(C_0/5) / k
    if input_c0 > 5.0:
        duration_hours = np.log(input_c0 / 5.0) / calculated_k
    else:
        duration_hours = 0.0
        
    st.info(f"🎯 **수학적 모델 예측 결과**: 현재 설정된 조건에서 타이레놀의 약효 지속 시간은 복용 후 약 **{duration_hours:.2f}시간** 입니다.")
    
    # 시뮬레이션 그래프 시각화
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(t_sim, c_sim, color="teal", linewidth=3, label="시간별 타이레놀 농도 예측 추이 C(t)")
    
    # 치료 범위 가이드선 표시
    ax3.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.5, label="최소 유효 농도 (MEC = 5.0)")
    ax3.axhline(y=25.0, color="red", linestyle=":", linewidth=1.5, label="최소 독성 농도 (MTC = 25.0)")
    
    # 약효 지속 구간 색칠하기
    ax3.fill_between(t_sim, 0, c_sim, where=(c_sim >= 5.0), color="orange", alpha=0.1, label="약효 유지 구간")
    
    ax3.set_xlabel("시간 (hours)")
    ax3.set_ylabel("타이레놀 혈중 농도 (ug/mL)")
    ax3.set_ylim(0, max(input_c0 + 5, 30))
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")
    st.pyplot(fig3)
    
    # 복용 가이드라인 판단 및 피드백
    st.subheader("🏥 AI 임상 가이드 및 복용 타이밍 조언")
    if input_c0 >= 25.0:
        st.error(f"🚨 **위험 (간독성 경고)**: 초기 농도 {input_c0} $\mu g/mL$는 약학정보원 기준 최소 독성 농도(25 $\mu g/mL$) 이상입니다. 과다 복용 위험이 있으므로 투여량을 즉시 줄여야 합니다.")
    elif duration_hours > 0:
        st.success(f"⏰ **추천 투약 주기**: 약효가 {duration_hours:.1f}시간 동안 유지된 후 소실됩니다. 안전한 치료 효과를 위해 약 **{max(4.0, np.round(duration_hours)):.0f}시간~6시간 간격**으로 추가 복용 타이밍을 설계하는 것이 타당합니다.")
        
    # 수학적 모델의 유용성과 한계 토론 세션 가이드
    st.markdown("---")
    st.subheader("💡 🤔 수학적 모델의 유용성과 한계 (4차시 토론 및 보고서 연계용)")
    with st.expander("이 시뮬레이션 모델이 가지는 '한계점'은 무엇일까요? (클릭하여 확인)"):
        st.markdown("""
        1. **흡수 단계의 생략**: 본 모델은 주사를 맞자마자 온몸에 즉시 퍼진다는 '정맥 주사(1구획 모델)'를 가정한 1계 미분방정식입니다. 하지만 우리가 입으로 먹는 타이레놀 정제는 위와 장에서 흡수되는 시간($T_{\text{max}} \approx 0.5\sim1$시간)이 걸리므로, 실제 초반 농도 상승 곡선은 반영하지 못한다는 수학적 한계가 있습니다.
        2. **개인차의 단순화**: 실제 환자의 대사 속도는 체중, 나이, 유전적 효소 활성도, 음식물 섭취 여부에 따라 시시각각 변하지만, 본 모델은 배설 상수 $k$를 고정된 상수로 취급하였습니다.
        3. **복합 대사 경로 무시**: 아세트아미노펜이 간에서 독성 물질(NAPQI)로 변환되어 글루타티온과 결합하는 화학적 포화 반응 메커니즘이 누락되어 있습니다.
        """)
