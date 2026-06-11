import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# 페이지 기본 설정 및 모던 디자인(CSS)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="약물 동태학 수학적 모델링", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f2937; margin-bottom: 0px; }
    .sub-text { font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem; }
    .step-title { font-size: 1.4rem; font-weight: 700; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 0.5rem; margin-top: 2rem;}
    .info-box { background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;}
    .question-text { font-weight: 600; color: #0284c7; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 변화의 수학적 모델링 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">데이터 관찰부터 세 가지 수학적 모델 비교, 잔차 분석, 그리고 역함수를 활용한 복용 주기 예측까지 주도적으로 수행해 봅시다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [데이터 생성] 가상의 시간-농도 데이터
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 30) 
C_0_true = 25.0
k_true = 0.35
noise = np.random.normal(0, 1.2, len(time_data))
conc_data = np.maximum(C_0_true * np.exp(-k_true * time_data) + noise, 0.1)
df_sample = pd.DataFrame({"Time(h)": np.round(time_data, 1), "Conc(mg/L)": np.round(conc_data, 2)})

# -----------------------------------------------------------------------------
# [모델 피팅 함수 및 R^2 계산] (백그라운드에서 미리 계산)
# -----------------------------------------------------------------------------
def get_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return max(0, 1 - (ss_res / ss_tot))

# 1. 직선(1차 함수) 모델
p_lin = np.polyfit(time_data, conc_data, 1)
y_lin = np.polyval(p_lin, time_data)
r2_lin = get_r2(conc_data, y_lin)

# 2. 2차 함수 모델
p_quad = np.polyfit(time_data, conc_data, 2)
y_quad = np.polyval(p_quad, time_data)
r2_quad = get_r2(conc_data, y_quad)

# 3. 지수 함수 모델
def exp_func(t, c0, k): return c0 * np.exp(-k * t)
p_exp, _ = curve_fit(exp_func, time_data, conc_data, p0=[25, 0.3])
y_exp = exp_func(time_data, *p_exp)
r2_exp = get_r2(conc_data, y_exp)

# -----------------------------------------------------------------------------
# 탭 구성: 1~2단계(데이터 및 탐구), 3단계(예측)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 1~2단계: 데이터 관찰 및 주도적 모델 탐구", "⏱️ 3단계: 공식 도출 및 복용 간격 예측"])

# =============================================================================
# TAB 1: 1단계 & 2단계 (학생 주도형)
# =============================================================================
with tab1:
    # --- 1단계: 데이터 관찰 ---
    st.markdown('<div class="step-title">1단계: 데이터 관찰 및 산점도</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.write("📋 **혈중 농도 데이터 (총 30개)**")
        st.dataframe(df_sample, height=250, use_container_width=True)
        
    with col2:
        st.write("📈 **파이썬 산점도 시각화**")
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.scatter(time_data, conc_data, color="#e11d48", alpha=0.8, label="측정 데이터")
        ax1.set_xlabel("시간 (h)", color="#4b5563")
        ax1.set_ylabel("농도 (mg/L)", color="#4b5563")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)
        
        st.markdown('<div class="info-box"><span class="question-text">🤔 모양 관찰 토론:</span> 점들의 분포가 어떤 형태를 띠고 있나요? 시간이 지날수록 감소폭이 어떻게 변하는지 관찰해 봅시다.</div>', unsafe_allow_html=True)

    # --- 2단계: 수학적 모델 직접 적용해보기 ---
    st.markdown('<div class="step-title">2단계: 어떤 수학적 모델이 가장 적합할까? (피팅 & 잔차 비교)</div>', unsafe_allow_html=True)
    st.write("아래 세 가지 함수 모델을 직접 클릭하여 적용해 보세요. 데이터와 곡선이 얼마나 일치하는지($R^2$), 그리고 **잔차(실제 데이터와 곡선의 차이)가 특정한 모양을 그리지 않고 무작위로 흩어지는지** 확인해야 합니다.")

    # 학생이 직접 모델을 선택하도록 구성
    selected_model = st.radio("🧪 시험해 볼 수학적 모델을 선택하세요:", 
                              ["선택 안함", "1. 직선 모델 (1차 함수)", "2. 2차 함수 모델", "3. 지수 함수 모델"], 
                              horizontal=True)

    if selected_model != "선택 안함":
        # 선택된 모델에 따라 변수 할당
        if "직선" in selected_model:
            model_name = "직선 모델 (C = at + b)"
            y_pred, r2_val, color_code = y_lin, r2_lin, "#2563eb"
            feedback = "잔차(아래 그래프)를 확인해 보세요. 가운데가 움푹 파인 'U자' 혹은 'V자' 형태의 체계적인 패턴이 보이나요? 패턴이 있다면 이 모델은 데이터를 완벽히 설명하지 못한다는 뜻입니다."
        elif "2차" in selected_model:
            model_name = "2차 함수 모델 (C = at² + bt + c)"
            y_pred, r2_val, color_code = y_quad, r2_quad, "#059669"
            feedback = "R² 값은 매우 높습니다! 하지만 잔차 그래프를 자세히 보세요. 끝부분(시간이 지날수록)에서 잔차가 다시 커지거나 특정 방향으로 쏠리는 경향이 있나요?"
        else:
            model_name = "지수 함수 모델 (C = C₀ e^{-kt})"
            y_pred, r2_val, color_code = y_exp, r2_exp, "#d97706"
            feedback = "R² 값도 높고, 잔차 그래프를 보면 0을 기준으로 특정한 모양 없이 위아래로 무작위(Random)로 흩어져 있습니다. 이는 남은 오차가 순수한 '자연적 측정 오차'뿐임을 의미합니다."

        st.markdown("---")
        st.markdown(f"### 🔍 {model_name} 분석 결과")
        
        # 모델 피팅 및 잔차 그래프를 상하로 배치
        fig_m, ax_m = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1) 피팅 그래프
        ax_m[0].scatter(time_data, conc_data, color="#e11d48", alpha=0.5, label="측정 데이터")
        ax_m[0].plot(time_data, y_pred, color=color_code, lw=2.5, label=f"예측 곡선 (R² = {r2_val:.3f})")
        ax_m[0].set_ylabel("농도 (mg/L)")
        ax_m[0].spines['top'].set_visible(False); ax_m[0].spines['right'].set_visible(False)
        ax_m[0].legend()
        ax_m[0].grid(True, linestyle="--", alpha=0.3)
        
        # 2) 잔차 그래프
        residuals = conc_data - y_pred
        ax_m[1].axhline(0, color='gray', linestyle='--', lw=1.5)
        ax_m[1].scatter(time_data, residuals, color=color_code, s=40, alpha=0.8)
        ax_m[1].set_xlabel("시간 (h)")
        ax_m[1].set_ylabel("잔차 (오차)")
        ax_m[1].spines['top'].set_visible(False); ax_m[1].spines['right'].set_visible(False)
        
        st.pyplot(fig_m)
        
        # 모델별 맞춤형 탐구 힌트 제공
        st.info(f"🧐 **탐구 힌트:** {feedback}")

    st.markdown("---")
    # 최종 결론 도출 영역 (학생이 열어보기 전까지 숨김)
    with st.expander("✅ [탐구 결론] 세 가지 모델을 모두 확인했나요? 최종 결론을 확인해 봅시다."):
        st.success("""
        **최적의 수학적 모델은 '지수 함수 모델'입니다.**
        
        * **수학/통계적 이유:** 단순히 결정계수($R^2$)가 높은 것뿐만 아니라, **'잔차(Residual)가 특정한 패턴 없이 0을 중심으로 무작위로 분포해야 한다'**는 회귀 분석의 핵심 조건을 가장 잘 만족하기 때문입니다. 직선이나 2차 함수는 잔차에 U자 등의 뚜렷한 패턴이 남아 있어 적합하지 않습니다.
        * **과학적 이유:** 체내에서 약물이 분해되는 속도는 '현재 몸에 남아있는 약물의 농도'에 비례합니다 ($\\frac{dC}{dt} = -kC$). 이를 수학적으로 적분하면 필연적으로 지수 함수($C = C_0 e^{-kt}$)가 도출됩니다.
        
        👉 이제 이 지수 함수 모델을 가지고 3단계 탭으로 넘어가 실제 복용 시간을 예측해 봅시다!
        """)

# =============================================================================
# TAB 2: 3단계 (역함수 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 공식 도출 및 복용 간격 예측</div>', unsafe_allow_html=True)
    st.write("앞서 2단계 탐구에서 가장 적합한 모델로 판명된 **지수함수 모델**을 바탕으로 환자의 맞춤형 복용 주기를 설계합니다.")
    
    # 파라미터 도출
    opt_c0, opt_k = p_exp[0], p_exp[1]
    half_life = np.log(2) / opt_k
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔑 1. 파라미터 도출")
        st.write(f"- **초기 농도 ($C_0$):** {opt_c0:.2f} mg/L")
        st.write(f"- **소실 속도 상수 ($k$):** {opt_k:.4f}")
        st.write(f"- **반감기 ($t_{{1/2}} = \\frac{{\\ln 2}}{{k}}$):** {half_life:.2f} 시간")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 2. 역함수 적용")
        st.write("목표 농도($C_{min}$)에 도달하는 시간 $t$를 구하기 위해 지수함수의 역함수(자연로그)를 적용합니다.")
        st.latex(r"C = C_0 e^{-kt} \quad \Rightarrow \quad t = \frac{-\ln(C/C_0)}{k}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 3. 최소 유효 농도 설정")
        target_c = st.slider("최소 유효 농도 ($C_{min}$)", 1.0, 10.0, 5.0, 0.5)
        st.write("약효가 떨어지기 전, 다시 약을 먹어야 하는 기준 농도를 직접 설정해 보세요.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 복용 간격 예측 계산
    if opt_c0 > target_c:
        pred_time = -np.log(target_c / opt_c0) / opt_k
    else:
        pred_time = 0.0
        
    st.markdown("---")
    st.markdown(f"### 💡 역함수 연산 예측 결과: 다음 약 복용 시점은 약 **{pred_time:.1f}시간** 뒤입니다.")
    
    # 시각화
    t_pred = np.linspace(0, 24, 200)
    c_pred = opt_c0 * np.exp(-opt_k * t_pred)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.plot(t_pred, c_pred, color="#0f766e", lw=2.5, label=f"모델 예측선: $C(t) = {opt_c0:.1f}e^{{-{opt_k:.3f}t}}$")
    
    # 타겟 농도 라인 및 점 표시
    ax2.axhline(target_c, color="#d97706", linestyle="--", label=f"최소 유효 농도 설정선 ($C_{min}$ = {target_c})")
    ax2.axvline(pred_time, color="#d97706", linestyle=":")
    ax2.scatter([pred_time], [target_c], color="red", zorder=5, s=100)
    
    ax2.text(pred_time + 0.3, target_c + 1, f"약효 소실 지점\n(t = {pred_time:.1f}h)", color="red", fontweight="bold")
    ax2.fill_between(t_pred, 0, c_pred, where=(c_pred >= target_c), color="#10b981", alpha=0.15, label="약효 유지 시간 구간")
    
    ax2.set_xlabel("시간 (hours)", color="#4b5563")
    ax2.set_ylabel("혈중 농도 (mg/L)", color="#4b5563")
    ax2.set_xlim(0, max(12, pred_time + 5))
    ax2.set_ylim(0, opt_c0 + 5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend()
    st.pyplot(fig2)
