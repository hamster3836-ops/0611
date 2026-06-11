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
# [데이터 생성] 가상의 시간-농도 데이터
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 30) 
C_0_true = 25.0
k_true = 0.35
noise = np.random.normal(0, 1.5, len(time_data))
conc_data = np.maximum(C_0_true * np.exp(-k_true * time_data) + noise, 0.1)

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
    
    # Plotly 기본 산점도
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(x=time_data, y=conc_data, mode='markers', marker=dict(size=8, color='#e11d48'), name='측정값'))
    fig_data.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="시간 (hours)", yaxis_title="농도 (mg/L)", plot_bgcolor="white")
    fig_data.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_data.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    st.plotly_chart(fig_data, use_container_width=True)

    st.markdown('<div class="step-title">2단계: 나만의 수학적 모델 만들기 (동적 파라미터 탐구)</div>', unsafe_allow_html=True)
    st.write("아래 세 가지 함수 중 하나를 선택하고 슬라이더를 조절해 점들에 가장 잘 맞는 곡선을 직접 만들어 보세요. **그래프 아래쪽의 '잔차(오차)'가 특정한 패턴(U자, V자 등) 없이 무작위로 흩어져야 완벽한 모델입니다!**")

    # 학생이 탐구할 모델 유형 선택
    model_type = st.radio("🧪 탐구할 수학적 함수를 선택하세요:", ["1차 함수 (직선)", "2차 함수 (포물선)", "지수 함수 (감소 곡선)"], horizontal=True)
    
    st.markdown("---")
    col_sliders, col_graph = st.columns([1.2, 2.5])
    
    # 선택한 모델에 따라 동적 슬라이더 제공 및 y_pred 계산
    time_trend = np.linspace(0, 12, 100) # 부드러운 곡선을 위한 시간 배열
    
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

        # R^2 및 잔차 계산
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
        # Plotly Subplots 생성 (위: 곡선 피팅, 아래: 잔차)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3], subplot_titles=("데이터와 모델 피팅", "잔차 (Residual) 분포"))
        
        # [위쪽 그래프] 데이터와 곡선
        fig.add_trace(go.Scatter(x=time_data, y=conc_data, mode='markers', name='측정값', marker=dict(color='#e11d48', size=7)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_trend, y=y_curve, mode='lines', name='내가 만든 모델', line=dict(color=color_theme, width=3)), row=1, col=1)
        
        # [아래쪽 그래프] 잔차 패턴
        fig.add_trace(go.Scatter(x=time_data, y=residuals, mode='markers', name='잔차(오차)', marker=dict(color=color_theme, size=6)), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
        
        # 레이아웃 디자인 설정
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", showlegend=False)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="농도 (mg/L)", row=1, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="시간 (hours)", row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="잔차", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("✅ [탐구 결론] 세 가지 함수를 모두 조작해 보았나요? 과학적 결론을 확인해 봅시다."):
        st.success("""
        **가장 완벽한 수학적 모델은 '지수 함수'입니다.**
        
        * **수학/통계적 이유:** 1차 함수(직선)나 2차 함수는 슬라이더를 아무리 잘 조절해 점수를 90점 이상으로 올려도, **아래쪽 잔차 그래프에 U자나 V자 모양의 뚜렷한 패턴**이 계속 남게 됩니다. 오직 **지수 함수만이 잔차를 0 주변에 무작위로 흩어지게** 만들 수 있습니다.
        * **과학적 이유:** 우리 몸에서 약물이 분해되는 속도는 '현재 몸에 남아있는 약물의 농도'에 비례($\\frac{dC}{dt} = -kC$)하기 때문에, 이를 풀면 필연적으로 지수함수 형태($C = C_0 e^{-kt}$)가 도출됩니다.
        """)

# =============================================================================
# TAB 2: 3단계 (역함수 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 공식 도출 및 복용 간격 예측</div>', unsafe_allow_html=True)
    st.write("탐구 결과 가장 적합했던 **지수함수 모델**을 바탕으로 맞춤형 복용 주기를 설계합니다. (가장 이상적인 파라미터 $C_0=25, k=0.35$ 를 사용합니다.)")
    
    # 3단계용 파라미터 (학생들이 찾은 가장 이상적인 값으로 고정)
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

    # 복용 간격 예측 계산
    if opt_c0 > target_c:
        pred_time = -np.log(target_c / opt_c0) / opt_k
    else:
        pred_time = 0.0
        
    st.markdown("---")
    st.markdown(f"### 💡 예측 결과: 다음 약 복용 시점은 약 **{pred_time:.1f}시간** 뒤입니다.")
    
    # Plotly 시뮬레이션 시각화
    t_pred = np.linspace(0, 24, 200)
    c_pred = opt_c0 * np.exp(-opt_k * t_pred)
    
    fig_pred = go.Figure()
    # 모델 예측선
    fig_pred.add_trace(go.Scatter(x=t_pred, y=c_pred, mode='lines', name='농도 예측 모델', line=dict(color='#0f766e', width=3)))
    # 최소 유효 농도 선
    fig_pred.add_hline(y=target_c, line_dash='dash', line_color='#d97706', annotation_text=f"최소 유효 농도 ({target_c})")
    # 약효 소실 지점(복용 시점) 점 표시
    fig_pred.add_trace(go.Scatter(x=[pred_time], y=[target_c], mode='markers+text', name='복용 시점',
                                  marker=dict(color='red', size=12), text=[f"약효 소실 지점 (t={pred_time:.1f}h)"], textposition="top right", textfont=dict(color="red", size=13)))
    
    fig_pred.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", showlegend=False)
    fig_pred.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="시간 (hours)", range=[0, 15])
    fig_pred.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="혈중 농도 (mg/L)", range=[0, 30])
    st.plotly_chart(fig_pred, use_container_width=True)
