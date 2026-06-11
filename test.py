import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 페이지 기본 설정 및 심플한 디자인(CSS) 적용
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pharmacokinetics Modeling", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f2937; margin-bottom: 0px; }
    .sub-text { font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem; }
    .step-title { font-size: 1.3rem; font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; margin-top: 1.5rem;}
    .highlight-box { background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 변화의 수학적 모델링 융합 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">데이터의 경향성을 관찰하고, 이를 설명할 수 있는 최적의 수학적 함수 모델을 직접 구축해 봅니다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [공통 데이터 생성] 2차시 탐구용 가상 환자 데이터셋
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 60)
C_0_true = 25.0
k_true = 0.35
noise = np.random.normal(0, 0.8, len(time_data))
conc_data = C_0_true * np.exp(-k_true * time_data) + noise
conc_data = np.maximum(conc_data, 0.1)
df_sample = pd.DataFrame({"Time_hours": time_data, "Concentration": conc_data})

# -----------------------------------------------------------------------------
# 사이드바 설정 (3차시용)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 예측 시뮬레이션 설정 (3차시)")
    st.write("모델이 완성된 후, 아래 파라미터를 변경하여 약효 지속 시간을 예측합니다.")
    input_c0 = st.slider("초기 투여 농도 (ug/mL)", 10.0, 50.0, 25.0)
    input_half_life = st.slider("환자의 약물 반감기 (시간)", 1.0, 6.0, 2.0)
    calculated_k = np.log(2) / input_half_life

# -----------------------------------------------------------------------------
# 메인 화면 - 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["[탐구 1] 데이터 경향성 파악 및 수학적 모델링", "[탐구 2] 완성된 모델을 활용한 처방 시뮬레이션"])

# -----------------------------------------------------------------------------
# 2차시 탭: 귀납적 함수식 탐구 (수식 은닉, 곡선 피팅 먼저)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="step-title">Step 1. 데이터 관찰 및 경향성 곡선 찾기</div>', unsafe_allow_html=True)
    st.write("수집된 실제 약물 농도 데이터(붉은 점)의 분포를 관찰하고, 슬라이더를 조절해 이 데이터들의 흐름을 가장 잘 나타내는 '최적의 곡선'을 그려보세요. (수식은 아직 생각하지 않아도 됩니다.)")
    
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.write("💡 **곡선 조작 패널**")
        # 수식(a, b) 대신 직관적인 설명 사용
        manual_a = st.slider("1️⃣ 곡선의 시작 높이", 10.0, 40.0, 15.0, step=0.5, help="시간이 0일 때의 위치를 조절합니다.")
        manual_b = st.slider("2️⃣ 곡선의 감소하는 모양 (휘어짐 정도)", 0.01, 1.00, 0.10, step=0.01, help="값이 커질수록 가파르게 떨어집니다.")
        
        # 일치도(Score) 계산
        user_curve = manual_a * np.exp(-manual_b * time_data)
        ss_res = np.sum((conc_data - user_curve)**2)
        ss_tot = np.sum((conc_data - np.mean(conc_data))**2)
        r2 = max(0, 1 - (ss_res / ss_tot))
        score = int(r2 * 100)
        
        st.write("---")
        st.metric(label="데이터와 곡선의 일치도", value=f"{score} %")
        
        # 심플한 피드백
        if score >= 95:
            st.success("훌륭합니다! 데이터의 경향을 완벽하게 반영하는 곡선을 찾았습니다.")
        elif score >= 80:
            st.info("거의 비슷해졌습니다. 미세하게 모양을 더 다듬어 보세요.")
        else:
            st.warning("아직 오차가 큽니다. 슬라이더를 더 움직여보세요.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 심플하고 세련된 그래프
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(time_data, conc_data, color="#e11d48", alpha=0.6, s=30, label="수집된 데이터 (측정값)")
        # 범례에서 수식 숨김
        ax.plot(time_data, user_curve, color="#2563eb", linewidth=2.5, label="내가 만든 경향성 곡선")
        
        ax.set_xlabel("Time (hours)", fontsize=10, color="#4b5563")
        ax.set_ylabel("Concentration (ug/mL)", fontsize=10, color="#4b5563")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        st.pyplot(fig)

    # 일치도가 95% 이상일 때 '수학적 모델링' 토론 파트 오픈
    if score >= 95:
        st.markdown('<div class="step-title">Step 2. 곡선의 수학적 모델링 (함수식 도출)</div>', unsafe_allow_html=True)
        st.write("데이터의 흐름을 지배하는 규칙을 시각적으로 찾아내었습니다. 그렇다면 우리가 방금 그린 이 곡선은 수학적으로 어떻게 표현할 수 있을까요?")
        
        st.info("🤔 **탐구 질문:** 우리가 조작한 곡선은 일정한 비율로 감소하며 x축에 점점 가까워지는 형태를 띱니다. 고등학교 수학에서 배운 함수 중 어떤 함수가 이 모델에 가장 적합할까요?")
        
        with st.expander("👉 수학적 모델 및 원리 확인하기"):
            st.markdown(f"""
            **1. 모델의 정체: 지수 함수 (Exponential Function)**
            * 우리가 찾은 곡선은 **$y = a \cdot e^{-bx}$** 형태의 지수 함수입니다.
            * 여러분이 방금 맞춘 최적의 파라미터를 대입하면, 이 데이터의 수학적 모델은 약 **$C(t) = {manual_a:.1f} \cdot e^{-{manual_b:.2f}t}$** 가 됩니다.
            
            **2. 왜 하필 지수 함수일까? (미적분과의 융합)**
            * 화학 반응이나 체내 약물 대사에서, 약물이 제거되는 '속도(변화율)'는 '현재 남아있는 약물의 농도'에 비례합니다.
            * 이를 미분방정식으로 표현하면 **$\\frac{{dC}}{{dt}} = -kC$** 가 되며, 이 방정식을 수학적으로 풀면 위와 같은 지수함수 모델이 자연스럽게 유도됩니다.
            """)

# -----------------------------------------------------------------------------
# 3차시 탭: 타이레놀 효과 및 투약 시간 예측 (심플/모던 디자인 유지)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="step-title">예측 모델 기반 맞춤형 투약 시뮬레이션</div>', unsafe_allow_html=True)
    st.write("앞서 수학적으로 증명한 '지수 감소 모델'을 바탕으로, 타이레놀의 안전한 복용 시간을 설계해 봅니다. (좌측 사이드바 패널을 조작하세요)")
    
    t_sim = np.linspace(0, 24, 300)
    c_sim = input_c0 * np.exp(-calculated_k * t_sim)
    
    if input_c0 > 5.0:
        duration_hours = np.log(input_c0 / 5.0) / calculated_k
    else:
        duration_hours = 0.0
        
    st.markdown(f"**💡 시뮬레이션 결과:** 현재 조건에서 약효 유지 시간은 약 **{duration_hours:.1f}시간**으로 계산되었습니다.")
    
    # 심플하고 가독성 높은 시뮬레이션 그래프
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(t_sim, c_sim, color="#0f766e", linewidth=2.5, label="혈중 약물 농도 예측 모델")
    
    # 치료 범위 가이드라인
    ax3.axhline(y=5.0, color="#d97706", linestyle="--", linewidth=1.5, label="최소 유효 농도 (5.0)")
    ax3.axhline(y=25.0, color="#be123c", linestyle=":", linewidth=1.5, label="최소 독성 농도 (25.0)")
    ax3.fill_between(t_sim, 0, c_sim, where=(c_sim >= 5.0)&(c_sim <= 25.0), color="#10b981", alpha=0.1)
    
    ax3.set_xlabel("Time (hours)", color="#4b5563")
    ax3.set_ylabel("Concentration (ug/mL)", color="#4b5563")
    ax3.set_ylim(0, max(input_c0 + 5, 30))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, loc="upper right")
    st.pyplot(fig)
    st.pyplot(fig3)
    
    # 임상 처방 피드백
    if input_c0 >= 25.0:
        st.error("⚠️ **간독성 위험:** 초기 농도가 최소 독성 농도(25.0)를 초과했습니다. 복용량을 줄여야 합니다.")
    elif duration_hours > 0:
        suggested_interval = max(4.0, np.round(duration_hours))
        st.success(f"✅ **안전 처방:** 약 {suggested_interval:.0f}시간 주기로 재복용하는 것이 안전하고 효과적입니다.")
    else:
        st.warning("⚠️ 복용량이 부족하여 약효가 나타나지 않습니다.")
