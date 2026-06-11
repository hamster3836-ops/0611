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
# 2차시 탭: 데이터 분포 시각화 및 탐구
# -----------------------------------------------------------------------------
with tab1:
    st.header("2차시: 약물 농도 감소는 어떤 수학적 모델로 설명할 수 있을까?")
    st.markdown("""
    **주요 활동**: 제시된 시간에 따른 약물 농도 데이터를 관찰하고, 점들의 분포를 
    가장 잘 대변하는 곡선을 직접 그려보며 수학적 함수 식을 탐구해 봅시다.
    """)
    
    # [단계 1] 데이터 표 격리 (클릭 시 열림)
    with st.expander("📋 [단계 1] 수집된 약물 농도 데이터 전체 보기 (클릭하여 열기)"):
        st.write("MBL 센서나 스마트 기기로 수집된 가상의 시간별 농도 데이터셋(총 60행)입니다.")
        st.dataframe(df_sample, use_container_width=True)
        
    st.markdown("---")
    
    # [단계 2] 시각화 및 탐구 곡선 그리기 (클릭 시 열리도록 수정)
    with st.expander("📊 [단계 2] 데이터 분포 시각화 및 곡선 그려보기 (클릭하여 열기)", expanded=True):
        st.write("아래 슬라이더를 움직여 붉은색 점(데이터)들의 흐름을 관통하는 나만의 곡선을 완성해 보세요.")

        # 학생들이 직접 그래프를 그리기 위한 파라미터 조절 슬라이더
        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            manual_a = col_sl1.slider("곡선의 시작 높이 조절 (파라미터 a)", 10.0, 40.0, 15.0, step=0.5)
        with col_sl2:
            manual_b = col_sl2.slider("곡선의 감소 속도 조절 (파라미터 b)", 0.01, 1.00, 0.10, step=0.01)

        # 그래프 시각화
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.scatter(df_sample["Time_hours"], df_sample["Concentration"], color="crimson", alpha=0.7, label="실제 측정 데이터 (오차 포함)")
        
        # 곡선 식 감추기 (라벨에서 수식 제거)
        time_trend = np.linspace(0, 12, 200)
        user_curve = manual_a * np.exp(-manual_b * time_trend)
        ax.plot(time_trend, user_curve, color="darkblue", linewidth=2.5, 
                 label="내가 찾은 경향성 곡선") # <- 수식(y=...)을 제거했습니다.
        
        ax.set_xlabel("시간 (Time, hours)")
        ax.set_ylabel("농도 (Concentration, ug/mL)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        st.pyplot(fig)

    st.markdown("---")
    
    # [단계 3] 수식 탐구 세션
    st.subheader("🧐 [단계 3] 함수 식 탐구 및 토론 과제")
    st.write("데이터의 흐름과 가장 잘 겹치도록 곡선을 그리셨나요? 그렇다면 이 곡선이 어떤 수학적 함수인지 유추해 봅시다.")
    
    st.info("""
    **Q. 우리가 방금 파라미터를 조절하여 그린 이 곡선은 고등학교에서 배운 어떤 함수 모델에 해당할까요?**
    * 1차 함수 ($y = ax + b$)
    * 2차 함수 ($y = ax^2 + bx + c$)
    * 지수 함수 ($y = a \\cdot b^{-x}$ 또는 $y = a \\cdot e^{-bx}$)
    * 로그 함수 ($y = a \\cdot \\ln(x) + b$)
    """)
    
    # 정답을 바로 보여주지 않고 클릭해야만 나타나도록 수정
    with st.expander("💡 힌트 및 정답 확인하기"):
        st.success("""
        **🧠 정답: 지수 함수 (Exponential Model)**
        * 우리가 그린 곡선은 대표적인 **지수 함수 감소(Exponential Decay)** 모델입니다.
        * 약동학에서 약물이 체내에서 제거되는 속도는 '현재 남아있는 농도'에 비례합니다. 이를 미분방정식($\\frac{dC}{dt} = -kC$)으로 세우고 풀면, 이와 같은 형태의 지수함수 수식($C(t) = C_0 \\cdot e^{-kt}$)이 도출됩니다.
        * 점들과 가장 완벽하게 일치할 때 파라미터는 대략 **a = 25.0, b = 0.35** 부근이 됩니다.
        """)

# -----------------------------------------------------------------------------
# 3차시 탭: 약학정보원 데이터 기반 타이레놀 복용 시간 예측 및 한계 설명
# -----------------------------------------------------------------------------
with tab2:
    st.header("3차시: 약물 농도 변화 모델을 활용하여 약물 효과 지속 시간 예측하기")
    st.subheader("📋 약학정보원(health.kr) 타이레놀(아세트아미노펜) 가이드라인 연계")
    
    st.markdown("""
    * **치료 농도 범위(Therapeutic Window)**: 약효를 나타내는 최소 유효 농도(**MEC**)는 **5 $\mu g/mL$**, 간독성 등 부작용을 유발하는 최소 독성 농도(**MTC**)는 **25 $\mu g/mL$**입니다.
    * **변인 통제 실습**: 왼쪽 사이드바의 슬라이더를 조절하여 초기 투여량($C_0$)과 환자의 반감기를 변화시켰을 때 약효 지속 시간이 어떻게 바뀌는지 관찰하세요.
    """)
    
    # 사이드바 입력값을 기반으로 미래 시점(24시간)까지 시뮬레이션 데이터 생성
    t_sim = np.linspace(0, 24, 300)
    c_sim = input_c0 * np.exp(-calculated_k * t_sim)
    
    # 약효 지속 시간 수학적 계산
    if input_c0 > 5.0:
        duration_hours = np.log(input_c0 / 5.0) / calculated_k
    else:
        duration_hours = 0.0
        
    st.info(f"🎯 **수학적 모델 시뮬레이션 결과**: 설정된 조건에서 타이레놀의 약효 지속 시간은 복용 후 약 **{duration_hours:.2f}시간** 입니다.")
    
    # 시뮬레이션 그래프 시각화
    fig3, ax3 = plt.subplots(figsize=(9, 4.2))
    ax3.plot(t_sim, c_sim, color="teal", linewidth=3, label="시간별 타이레놀 예측 농도 C(t)")
    ax3.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.5, label="최소 유효 농도 (MEC = 5.0)")
    ax3.axhline(y=25.0, color="red", linestyle=":", linewidth=1.5, label="최소 독성 농도 (MTC = 25.0)")
    ax3.fill_between(t_sim, 0, c_sim, where=(c_sim >= 5.0), color="orange", alpha=0.1, label="약효 유효 영역")
    ax3.set_xlabel("시간 (hours)")
    ax3.set_ylabel("타이레놀 혈중 농도 (ug/mL)")
    ax3.set_ylim(0, max(input_c0 + 5, 32))
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")
    st.pyplot(fig3)
    
    # 복용 타이밍 가이드라인 조언
    st.subheader("🏥 AI 기반 임상 처방 조언 및 투약 타이밍 설계")
    if input_c0 >= 25.0:
        st.error(f"🚨 **위험 (간독성 유발 가능성)**: 설정한 초기 농도 {input_c0} $\mu g/mL$는 약학정보원 기준 최소 독성 농도(25 $\mu g/mL$) 이상입니다! 과다 복용 위험이 있으므로 투여 용량을 낮추어야 합니다.")
    elif duration_hours > 0:
        suggested_interval = max(4.0, np.round(duration_hours))
        st.success(f"⏰ **추천 추가 복용 주기**: 약효가 {duration_hours:.1f}시간 동안 유지된 후 유효 농도 아래로 떨어집니다. 안전하고 지속적인 진통 효과를 위해 약 **{suggested_interval:.0f}시간 ~ 6시간 간격**으로 재복용 시간을 설계하는 것이 수학적으로 적절합니다.")
    else:
        st.warning("⚠️ 최초 투여 농도가 너무 낮아 최소 유효 농도(5 ㎍/mL)에 도달하지 못해 약효가 나타나지 않습니다.")
        
    # 수학적 모델의 유용성과 한계 토론 세션 가이드
    st.markdown("---")
    st.subheader("💡 🤔 수학적 모델의 유용성과 한계 설명하기 (3차시 정리용)")
    with st.expander("생기부 기록 및 보고서 작성을 위한 질문: 이 시뮬레이션 모델이 가지는 한계점은 무엇일까요?"):
        st.markdown("""
        1. **경구 흡수 과정의 생략**: 본 모델은 투여 즉시 혈액에 100% 퍼지는 '정맥 주사' 모델을 기반으로 삼고 있습니다. 실제 입으로 먹는 타이레놀 알약은 위와 장에서 흡수되어 혈중 농도가 최고점에 도달할 때까지 **약 30분 ~ 1시간($T_{\text{max}}$)의 상승 구간**이 존재하지만, 이 모델은 이를 반영하지 못하는 수학적 한계가 있습니다.
        2. **개인별 대사의 정형화**: 사람의 약물 대사 속도는 나이, 체중, 성별, 간 대사 효소(CYP2E1)의 활성도에 따라 달라지지만 본 시뮬레이션은 $k$를 고정된 상수로 단순화했습니다.
        3. **약물 축적 효과의 미반영**: 약을 여러 번 반복해서 먹었을 때 체내에 남아있는 잔여량과 새로 복용하는 양이 결합하여 누적되는 정밀한 수학적 모델(등비수열의 합)이 추가로 고려되어야 더 정확한 복용 알고리즘을 짤 수 있습니다.
        """)
