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
    .mission-box { background-color: #fdf4ff; padding: 1.5rem; border-radius: 8px; border: 1px solid #f5d0fe; margin-bottom: 1rem;}
    .explore-box { background-color: #fffbeb; padding: 1.5rem; border-radius: 8px; border: 1px solid #fde68a; margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 수학적 모델링 융합 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">데이터를 관찰하고, 나만의 수학적 모델을 구축한 뒤, 이를 이용해 실제 복용 시간을 예측해 봅시다.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# [데이터 고정] 캐싱
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

# -----------------------------------------------------------------------------
# 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🕵️‍♂️ 1~2단계: 수학적 모델 동적 탐구", "⏱️ 3단계: 모델 완성 및 복용 간격 예측"])

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
    st.write("아래 세 가지 함수 중 하나를 선택하고 슬라이더를 조절해 점들에 가장 잘 맞는 곡선을 직접 만들어 보세요.")

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

    st.markdown('<div class="explore-box">', unsafe_allow_html=True)
    st.markdown("### 🤔 [심화 탐구] 어떤 수학적 모델이 가장 적합할까요?")
    st.markdown("""
    세 가지 함수의 파라미터를 모두 조작해 보았나요? 단순히 눈으로 보기에 곡선이 비슷하다고 해서 올바른 과학적 모델인 것은 아닙니다. 조원들과 함께 아래의 두 가지 관점에서 데이터를 분석하고 진짜 정답을 찾아보세요!

    **1. 통계적 관점: '잔차(Residual)'에 숨겨진 단서**
    * 1차 함수나 2차 함수 모델은 점수를 아무리 올려도 아래쪽 잔차 그래프에 U자나 V자 모양의 일정한 '패턴'이 남게 됩니다. 잔차가 가장 무작위(Random)로 흩어지는 모델은 무엇인가요?

    **2. 과학 및 수학적 관점: 약물이 분해되는 속도 (변화율)**
    * 실제 우리 몸의 간은 **'현재 몸에 남아있는 약물의 양(농도)'에 비례**하여 약물을 분해합니다.
    * 💡 **탐구 질문:** '약물 농도의 변화율(속도)이 현재 농도 $C$에 비례한다'는 조건을 수식으로 표현하면 $\\frac{dC}{dt} = -kC$ 가 됩니다. 이 미분방정식을 만족하는 함수는 1차, 2차, 지수 함수 중 무엇일까요?
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: 3단계 (수식 도출 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 수식 도출 및 맞춤형 복용 주기 예측</div>', unsafe_allow_html=True)
    
    st.write("1~2단계 탐구를 통해 가장 적합한 모델이 **지수함수 모델**임을 알아냈습니다. 이제 여러분이 찾은 파라미터를 이용해 모델을 완성하고 시뮬레이션을 진행해 봅시다.")
    
    # 💡 [미션 1] 수학적 모델 직접 완성하기
    st.markdown("### 🧩 [미션 1] 나만의 지수함수 모델 완성하기")
    st.markdown('<div class="mission-box">', unsafe_allow_html=True)
    st.write("1~2단계에서 오차가 가장 적고 잔차가 무작위로 퍼졌던 '초기 농도($C_0$)'와 '소실 속도 상수($k$)' 값을 아래에 입력해 보세요.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        student_c0 = st.number_input("초기 농도 ($C_0$) 입력:", min_value=0.0, max_value=50.0, step=0.5, value=0.0)
    with col_input2:
        student_k = st.number_input("소실 속도 상수 ($k$) 입력:", min_value=0.00, max_value=1.00, step=0.01, value=0.00)
    st.markdown('</div>', unsafe_allow_html=True)

    is_model_correct = False
    if student_c0 > 0 and student_k > 0:
        # 학생이 찾은 파라미터가 이상적인 값(25.0, 0.35) 근처인지 확인
        if abs(student_c0 - 25.0) <= 1.5 and abs(student_k - 0.35) <= 0.05:
            st.success("🎉 정확합니다! 이상적인 파라미터를 아주 잘 찾았군요. 아래의 환자 맞춤형 모델이 완성되었습니다.")
            st.latex(rf"C(t) = {student_c0:.1f} \cdot e^{{-{student_k:.2f}t}}")
            is_model_correct = True
        else:
            st.error("앗, 값이 조금 다릅니다. 1~2단계에서 R² 점수가 가장 높았던 지수함수의 슬라이더 값을 다시 확인해 보세요. (힌트: $C_0$는 25 부근, $k$는 0.35 부근입니다.)")

    # 💡 [미션 2] 반감기 스스로 계산하기
    if is_model_correct:
        st.markdown("---")
        st.markdown("### ✍️ [미션 2] 약물 반감기($t_{1/2}$) 직접 계산하기")
        
        col_m2, col_m2_hint = st.columns([1.2, 1])
        with col_m2:
            st.markdown('<div class="mission-box">', unsafe_allow_html=True)
            st.write("반감기는 약물 농도가 처음($C_0$)의 정확히 절반($C_0 / 2$)으로 줄어드는 시간입니다. 방금 완성한 수식을 이용해 환자의 반감기를 계산해 보세요.")
            student_hl = st.number_input("계산한 반감기를 입력하세요 (소수 첫째 자리 권장):", min_value=0.0, max_value=15.0, step=0.1, value=0.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m2_hint:
            st.info(r"""
            **💡 반감기 계산 힌트**
            1. 완성된 식에 반감기 조건을 대입합니다: 
               $\frac{1}{2} C_0 = C_0 \cdot e^{-kt}$
            2. 양변을 $C_0$로 나누면: $\frac{1}{2} = e^{-kt}$
            3. 양변에 자연로그($\ln$)를 취하여 $t$값을 구해보세요. 
               (단, $\ln 2 \approx 0.693$ 으로 계산합니다.)
            """)

        is_hl_correct = False
        true_hl = np.log(2) / student_k
        if student_hl > 0:
            if abs(student_hl - true_hl) <= 0.2: # 오차 허용 범위 설정
                st.success(f"🎉 훌륭합니다! 환자의 반감기는 약 **{student_hl:.1f}시간** 입니다. 시뮬레이션 단계가 활성화되었습니다.")
                is_hl_correct = True
            else:
                st.error("다시 한 번 계산해 보세요. (힌트: 0.693을 방금 찾은 소실 속도 상수 k로 나누어 보세요.)")

        # 💡 [탐구 활동] 최소 유효 농도 설정 및 예측
        if is_hl_correct:
            st.markdown("---")
            st.markdown("### 🎯 [탐구 활동] 역함수 도출 및 복용 주기 시뮬레이션")
            st.write("안전한 복용 주기를 알려면, 약물이 목표하는 농도($C$)로 떨어지는 데 걸리는 시간($t$)을 구하는 **역함수**가 필요합니다.")
            st.latex(r"t = \frac{-\ln(C / C_0)}{k}")
            
            st.write("약효가 떨어지기 전, 다음 약을 복용해야 하는 기준선인 **최소 유효 농도($C_{min}$)**를 설정해 보세요.")
            
            target_c = st.slider("최소 유효 농도 설정 (단위: mg/L)", 1.0, 10.0, 5.0, 0.5)
            
            # 학생이 입력했던 C0, k를 바탕으로 예측 시간 계산
            pred_time = -np.log(target_c / student_c0) / student_k
            
            st.markdown(f"### ⏱️ 예측 결과: 다음 약 복용 시점은 투여 후 약 **{pred_time:.1f}시간** 뒤입니다.")
            
            # Plotly 시뮬레이션 시각화
            t_pred = np.linspace(0, 24, 200)
            c_pred = student_c0 * np.exp(-student_k * t_pred)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=t_pred, y=c_pred, mode='lines', name='농도 예측 모델', line=dict(color='#0f766e', width=3)))
            fig_pred.add_hline(y=target_c, line_dash='dash', line_color='#d97706', annotation_text=f"최소 유효 농도 ({target_c})")
            fig_pred.add_trace(go.Scatter(x=[pred_time], y=[target_c], mode='markers+text', name='복용 시점',
                                          marker=dict(color='red', size=12), text=[f"다음 복용 시점 (t={pred_time:.1f}h)"], textposition="top right", textfont=dict(color="red", size=13)))
            
            fig_pred.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", showlegend=False)
            fig_pred.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="시간 (hours)", range=[0, 15], fixedrange=True)
            fig_pred.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_text="혈중 농도 (mg/L)", range=[0, 30], fixedrange=True)
            st.plotly_chart(fig_pred, use_container_width=True)
