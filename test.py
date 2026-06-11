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
    .info-box { background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 변화의 수학적 모델링 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">데이터 관찰부터 세 가지 수학적 모델 비교, 잔차 분석, 그리고 역함수를 활용한 복용 주기 예측까지 단계별로 수행합니다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [데이터 생성] 가상의 시간-농도 데이터
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 30) # 명확한 잔차 확인을 위해 데이터 수 30개로 조정
C_0_true = 25.0
k_true = 0.35
noise = np.random.normal(0, 1.2, len(time_data))
conc_data = np.maximum(C_0_true * np.exp(-k_true * time_data) + noise, 0.1)
df_sample = pd.DataFrame({"Time(h)": np.round(time_data, 1), "Conc(mg/L)": np.round(conc_data, 2)})

# -----------------------------------------------------------------------------
# [모델 피팅 함수 및 R^2 계산]
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
# 탭 구성: 1~2단계(데이터 및 모델 비교), 3단계(예측)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 1~2단계: 데이터 관찰 및 수학적 모델 피팅", "⏱️ 3단계: 공식 도출 및 복용 간격 예측"])

# =============================================================================
# TAB 1: 1단계 & 2단계
# =============================================================================
with tab1:
    st.markdown('<div class="step-title">1단계: 데이터 관찰 및 산점도</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.write("📋 **혈중 농도 데이터**")
        st.dataframe(df_sample, height=250, use_container_width=True)
        
    with col2:
        st.write("📈 **파이썬 산점도 관찰**")
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.scatter(time_data, conc_data, color="#e11d48", alpha=0.8, label="측정 데이터")
        ax1.set_xlabel("시간 (h)", color="#4b5563")
        ax1.set_ylabel("농도 (mg/L)", color="#4b5563")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)
        
        st.info("🤔 **모양 관찰 토론:** 점들의 분포가 어떤 형태를 띠고 있나요? 직선일까요, 2차 곡선일까요, 아니면 다른 형태일까요?")

    st.markdown('<div class="step-title">2단계: 세 모델 피팅 및 잔차 비교</div>', unsafe_allow_html=True)
    st.write("데이터를 설명할 수 있는 세 가지 수학적 모델을 적용해 보고, $R^2$(결정계수)와 **잔차(실제값-예측값) 분포**를 통해 최적의 모델을 선택합니다.")
    
    # 세 가지 모델 시각화 컬럼
    m_col1, m_col2, m_col3 = st.columns(3)
    
    # [직선 모델]
    with m_col1:
        st.markdown("**1. 직선 모델 ($C = at + b$)**")
        fig_l, ax_l = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [2, 1]})
        ax_l[0].scatter(time_data, conc_data, color="#e11d48", alpha=0.5, s=20)
        ax_l[0].plot(time_data, y_lin, color="#2563eb", lw=2)
        ax_l[0].set_title(f"R² = {r2_lin:.3f}", fontsize=10)
        ax_l[0].axis('off')
        
        # 잔차 그래프
        ax_l[1].axhline(0, color='gray', linestyle='--', lw=1)
        ax_l[1].scatter(time_data, conc_data - y_lin, color="#2563eb")
        ax_l[1].set_title("잔차(Residual) 패턴", fontsize=9)
        ax_l[1].spines['top'].set_visible(False); ax_l[1].spines['right'].set_visible(False)
        st.pyplot(fig_l)
        st.caption("U자형 잔차 패턴이 나타남 (체계적 오차 발생)")

    # [2차 함수 모델]
    with m_col2:
        st.markdown("**2. 2차 함수 모델 ($C = at^2 + bt + c$)**")
        fig_q, ax_q = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [2, 1]})
        ax_q[0].scatter(time_data, conc_data, color="#e11d48", alpha=0.5, s=20)
        ax_q[0].plot(time_data, y_quad, color="#059669", lw=2)
        ax_q[0].set_title(f"R² = {r2_quad:.3f}", fontsize=10)
        ax_q[0].axis('off')
        
        ax_q[1].axhline(0, color='gray', linestyle='--', lw=1)
        ax_q[1].scatter(time_data, conc_data - y_quad, color="#059669")
        ax_q[1].set_title("잔차(Residual) 패턴", fontsize=9)
        ax_q[1].spines['top'].set_visible(False); ax_q[1].spines['right'].set_visible(False)
        st.pyplot(fig_q)
        st.caption("R²는 높으나, 특정 구간에서 쏠림 현상 존재")

    # [지수 함수 모델]
    with m_col3:
        st.markdown("**3. 지수 함수 모델 ($C = C_0 e^{-kt}$) ★**")
        fig_e, ax_e = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [2, 1]})
        ax_e[0].scatter(time_data, conc_data, color="#e11d48", alpha=0.5, s=20)
        ax_e[0].plot(time_data, y_exp, color="#d97706", lw=2)
        ax_e[0].set_title(f"R² = {r2_exp:.3f}", fontsize=10)
        ax_e[0].axis('off')
        
        ax_e[1].axhline(0, color='gray', linestyle='--', lw=1)
        ax_e[1].scatter(time_data, conc_data - y_exp, color="#d97706")
        ax_e[1].set_title("잔차(Residual) 패턴", fontsize=9)
        ax_e[1].spines['top'].set_visible(False); ax_e[1].spines['right'].set_visible(False)
        st.pyplot(fig_e)
        st.caption("잔차가 0을 중심으로 무작위로 분포함 (최적 모델)")

# =============================================================================
# TAB 2: 3단계
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 공식 도출 및 복용 간격 예측</div>', unsafe_allow_html=True)
    
    # 파라미터 도출
    opt_c0, opt_k = p_exp[0], p_exp[1]
    half_life = np.log(2) / opt_k
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔑 파라미터 도출")
        st.write(f"- **$C_0$ (초기 농도):** {opt_c0:.2f} mg/L")
        st.write(f"- **$k$ (소실 속도 상수):** {opt_k:.4f}")
        st.write(f"- **반감기 ($t_{{1/2}} = \\frac{{\\ln 2}}{{k}}$):** {half_life:.2f} 시간")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 역함수 적용")
        st.write("목표 농도($C_{min}$)에 도달하는 시간 $t$를 구하기 위해 지수함수의 역함수(자연로그)를 적용합니다.")
        st.latex(r"C = C_0 e^{-kt}")
        st.latex(r"\Rightarrow t = \frac{-\ln(C/C_0)}{k}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 최소 유효 농도 설정")
        target_c = st.slider("최소 유효 농도 ($C_{min}$)", 1.0, 10.0, 5.0, 0.5)
        st.write("약효가 떨어지기 전, 다시 약을 먹어야 하는 기준 농도입니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 복용 간격 예측 계산
    if opt_c0 > target_c:
        pred_time = -np.log(target_c / opt_c0) / opt_k
    else:
        pred_time = 0.0
        
    st.markdown("---")
    st.markdown(f"### 💡 예측 결과: 다음 복용 주기는 **약 {pred_time:.1f}시간** 뒤입니다.")
    
    # 시각화
    t_pred = np.linspace(0, 24, 200)
    c_pred = opt_c0 * np.exp(-opt_k * t_pred)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_pred, c_pred, color="#0f766e", lw=2.5, label=f"모델 예측선: $C(t) = {opt_c0:.1f}e^{{-{opt_k:.3f}t}}$")
    
    # 타겟 농도 라인 및 점 표시
    ax2.axhline(target_c, color="#d97706", linestyle="--", label=f"최소 유효 농도 ($C_{min}$ = {target_c})")
    ax2.axvline(pred_time, color="#d97706", linestyle=":")
    ax2.scatter([pred_time], [target_c], color="red", zorder=5, s=100)
    
    ax2.text(pred_time + 0.3, target_c + 1, f"약효 소실 지점\n(t = {pred_time:.1f}h)", color="red", fontweight="bold")
    ax2.fill_between(t_pred, 0, c_pred, where=(c_pred >= target_c), color="#10b981", alpha=0.15, label="약효 유지 시간")
    
    ax2.set_xlabel("시간 (hours)")
    ax2.set_ylabel("농도 (mg/L)")
    ax2.set_xlim(0, max(12, pred_time + 5))
    ax2.set_ylim(0, opt_c0 + 5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend()
    st.pyplot(fig2)
