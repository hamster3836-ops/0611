import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 페이지 기본 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(page_title="약물 동태학 동적 탐구", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0px; }
    .sub-text { font-size: 1.1rem; color: #475569; margin-bottom: 2rem; }
    .step-title { font-size: 1.4rem; font-weight: 700; color: #0f766e; border-bottom: 2px solid #ccfbf1; padding-bottom: 0.5rem; margin-top: 2rem;}
    .info-box { background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 수학적 모델링 동적 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">슬라이더를 움직여 파라미터를 조절하며, 잔차(오차) 패턴을 분석해 최적의 함수를 찾아봅시다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [데이터 고정] 캐싱을 통해 데이터가 매번 새로 생성되어 미세하게 바뀌는 현상 방지
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    t = np.linspace(0, 12, 30) 
    C_0_true = 25.0
    k_true = 0.35
    noise = np.random.normal(0, 1.5, len(t))
    c = np.maximum(C_0_true * np.exp(-k_true * t) + noise, 0.1)
    return t, c

time_data, conc_data = load_data()
df_sample = pd.DataFrame({"Time(h)": np.round(time_data, 1), "Conc(mg/L)": np.round(conc_data, 2)})

# -----------------------------------------------------------------------------
# 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🕵️‍♂️ 1~2단계: 수학적 모델 동적 탐구", "⏱️ 3단계: 공식 도출 및 복용 간격 예측"])

# =============================================================================
# TAB 1: 1단계 & 2단계 (동적 파라미터 조절)
# =============================================================================
with tab1:
    st.markdown('<div class="step-title">1단계: 측정 데이터 관찰</div>', unsafe_allow_html=True)
    st.write("혈중 약물 농도 측정 데이터가 어떻게 분포되어 있는지 확인합니다.")
    
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(x=time_data, y=conc_data, mode='markers', marker=dict(size=8, color='#e11d48'), name='측정값'))
    fig_data.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="시간 (hours)", yaxis_title="농도 (mg/L)", plot_bgcolor="white")
    fig_data.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-0.5, 12.5], autorange=False, fixedrange=True)
    fig_data.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-5, 45], autorange=False, fixedrange=True)
    st.plotly_chart(fig_data, use_container_width=True)

    st.markdown('<div class="step-title">2단계: 나만의 수학적 모델 만들기 (동적 파라미터 탐구)</div>', unsafe_allow_html=True)
    st.write("아래 세 가지 함수 중 하나를 선택하고 슬라이더를 조절해 점들에 가장 잘 맞는 곡선을 직접 만들어 보세요. **그래프 아래쪽의 '잔차(오차)'가 특정한 패턴(U자, V자 등) 없이 무작위로 흩어져야 완벽한 모델입니다!**")

    model_type = st.radio("🧪 탐구할 수학적 함수를 선택하세요:", ["1차 함수 (직선)", "2차 함수 (포물선)", "지수 함수 (감소 곡선)"], horizontal=True)
    
    st.markdown("---")
    col_sliders, col_graph = st.columns([1.2, 2.5])
    
    time_trend = np.linspace(0, 12, 100) 
    
    with col_sliders:
        st.markdown(f"#### 🎛️ {model_type} 파라미터 조절")
        
        if "1차" in model_type:
            st.latex(r"C(t) = a \cdot t + b")
            a_lin = st.slider("기울기 (a)", -5.0, 0.0, -2.0, 0.1)
            b_lin = st.slider("Y절편 (b)", 10.0, 40.0, 20.0, 0.5)
            y_pred = a_lin * time_data + b_lin
            y_curve = a_lin * time_trend + b_lin
            color_theme = "#2563eb"
            
        elif "2차" in model_type:
            st.latex(r"C(t) = a \cdot t^2 + b \cdot t + c")
            a_quad = st.slider("이차항 계수 (a)", -1.0, 2.0, 0.2, 0.05)
            b_quad = st.slider("일차항 계수 (b)", -10.0, 0.0, -3.0, 0.1)
            c_quad = st.slider("상수항 (c)", 10.0, 40.0, 25.0, 0.5)
            y_pred = a_quad * (time_data**2) + b_quad * time_data + c_quad
            y_curve = a_quad * (time_trend**2) + b_quad * time_trend + c_quad
            color_theme = "#059669"
            
        else: # 지수함수
            st.latex(r"C(t) = C_0 \cdot e^{-k \cdot t}")
            c0_exp = st.slider("초기 농도 (C₀)", 10.0, 40.0, 15.0, 0.5)
            k_exp = st.slider("소실 속도 상수 (k)", 0.01, 1.00, 0.10, 0.01)
            y_pred = c0_exp * np.exp(-k_exp * time_data)
            y_curve = c0_exp * np.exp(-k_exp * time_trend)
            color_theme = "#d97706"

        residuals = conc_data - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((conc_data - np.mean(conc_data))**2)
        r2 = max(0, 1 - (ss_res / ss_tot))
        
        st.markdown("---")
        st.metric(label="현재 모델의 일치도 (R² 점수)", value=f"{int(r2 * 100)} 점")
        if r2 > 0.9:
            st.success("점수가 매우 높습니다! 하지만 우측 하단의 **잔차 패턴**도 꼭 확인하세요.")
        else:
            st.warning("점수가 낮습니다. 슬라이더를 움직여 곡선을 점에 맞춰보세요.")

    with col_graph:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3], subplot_titles=("데이터와 모델 피팅", "잔차 (Residual) 분포"))
        
        fig.add_trace(go.Scatter(x=time_data, y=conc_data, mode='markers', name='측정값', marker=dict(color='#e11d48', size=7)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_trend, y=y_curve, mode='lines', name='내가 만든 모델', line=dict(color=color_theme, width=3)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=time_data, y=residuals, mode='markers', name='잔차(오차)', marker=dict(color=color_theme, size=6)), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", showlegend=False)
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-0.5, 12.5], autorange=False, fixedrange=True, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="농도 (mg/L)", range=[-5, 45], autorange=False, fixedrange=True, row=1, col=1)
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="시간 (hours)", range=[-0.5, 12.5], autorange=False, fixedrange=True, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="잔차", range=[-30, 30], autorange=False, fixedrange=True, row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    # 💡 [변경 포인트] 정답을 바로 주지 않고 토론과 탐구를 유도하는 질문형 구성
    with st.expander("🤔 [심화 탐구] 어떤 함수 모델이 가장 적합할까요? 스스로 증명해 봅시다!"):
        st.markdown("""
        단순히 곡선 모양이 비슷하다고 해서 올바른 과학적 모델인 것은 아닙니다. 아래의 두 가지 관점에서 조원들과 함께 토론해 보세요.

        **1. 통계적 관점: 잔차(Residual)의 숨겨진 의미**
        * 1차 함수(직선)나 2차 함수(포물선) 모델의 파라미터를 아무리 잘 조절해도, 모델 일치도($R^2$) 점수를 어느 정도까지만 올릴 수 있었습니다.
        * 점수가 높더라도 아래쪽 **잔차(Residual) 그래프**를 비교해 보세요. 1차와 2차 함수는 잔차가 U자나 V자 모양으로 일정한 '패턴'을 보입니다. 반면 지수 함수는 어떤가요? 
        * 💡 **탐구 질문:** 완벽하게 데이터를 설명하는 모델이라면, 모델이 잡아내지 못한 오차(잔차)는 일정한 규칙을 가져야 할까요, 아니면 0을 중심으로 무작위(Random)로 흩어져야 할까요?

        **2. 과학 및 수학적 관점: 약물이 사라지는 속도 (미적분)**
        * 1차 함수(직선) 모델은 "시간이 지나도 약물이 줄어드는 *절대적인 양*이 매시간 똑같다"는 것을 의미합니다. 과연 우리 몸의 대사 작용이 그럴까요?
        * 실제로 우리 몸에서 약물이 분해되는 속도는 **'현재 몸에 남아있는 약물의 양(농도)'에 비례**합니다. 몸에 약이 많으면 간이 열심히 일해서 빨리 분해하고, 약이 조금 남으면 천천히 분해하죠.
        * 💡 **탐구 질문:** '약물 농도의 변화율(속도)이 현재 농도 $C$에 비례한다'는 문장을 수학적인 기호로 표현하면 $\\frac{dC}{dt} = -kC$ 가 됩니다. 고등학교 수학 지식을 활용했을 때, 이 미분방정식을 만족하는 함수는 1차, 2차, 지수 함수 중 무엇이 되어야만 할까요?
        """)

# =============================================================================
# TAB 2: 3단계 (역함수 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 공식 도출 및 복용 간격 예측</div>', unsafe_allow_html=True)
    st.write("탐구 결과 가장 적합했던 **지수함수 모델**을 바탕으로 맞춤형 복용 주기를 설계합니다. (가장 이상적인 파라미터 $C_0=25, k=0.35$ 를 사용합니다.)")
    
    opt_c0, opt_k = 25.0, 0.35
    half_life = np.log(2) / opt_k
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔑 1. 이상적 파라미터")
        st.write(f"- **초기 농도 ($C_0$):** {opt_c0:.1f} mg/L")
        st.write(f"- **소실 상수 ($k$):** {opt_k:.2f}")
        st.write(f"- **반감기 ($t_{{1/2}}$):** 약 {half_life:.1f} 시간")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 2. 역함수 적용")
        st.write("목표 농도($C_{min}$)에 도달하는 시간 $t$를 구하기 위해 지수함수의 역함수(자연로그)를 적용합니다.")
        st.latex(r"t = \frac{-\ln(C/C_0)}{k}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 3. 최소 유효 농도 설정")
        target_c = st.slider("최소 유효 농도 ($C_{min}$)", 1.0, 10.0, 5.0, 0.5)
        st.write("약효가 떨어지기 전, 약을 다시 먹어야 하는 기준선입니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    if opt_c0 > target_c:
        pred_time = -np.log(target_c / opt_c0) / opt_k
    else:
        pred_time = 0.0
        
    st.markdown("---")
    st.markdown(f"### 💡 예측 결과: 다음 약 복용 시점은 약 **{pred_time:.1f}시간** 뒤입니다.")
    
    t_pred = np.linspace(0, 24, 200)
    c_pred = opt_c0 * np.exp(-opt_k * t_pred)
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=t_pred, y=c_pred, mode='lines', name='농도 예측 모델', line=dict(color='#0f766e', width=3)))
    fig_pred.add_hline(y=target_c, line_dash='dash', line_color='#d97706', annotation_text=f"최소 유효 농도 ({target_c})")
    fig_pred.add_trace(go.Scatter(x=[pred_time], y=[target_c], mode='markers+text', name='복용 시점',
                                  marker=dict(color='red', size=12), text=[f"약효 소실 지점 (t={pred_time:.1f}h)"], textposition="top right", textfont=dict(color="red", size=13)))
    
    fig_pred.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", showlegend=False)
    fig_pred.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="시간 (hours)", range=[0, 15], fixedrange=True)
    fig_pred.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="혈중 농도 (mg/L)", range=[0, 30], fixedrange=True)
    st.plotly_chart(fig_pred, use_container_width=True)
