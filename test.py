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
    .quiz-box { background-color: #f0fdf4; padding: 1.5rem; border-radius: 8px; border: 1px solid #bbf7d0; margin-bottom: 1rem;}
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

    # 💡 [직관적 식 추측 가이드] 미분방정식 없이 지수함수 도출하기
    with st.expander("🤔 [수학적 모델링] 미분방정식을 몰라도 이 식을 추측할 수 있을까요?"):
        st.markdown("""
        어려운 대학 미적분을 쓰지 않더라도, 우리는 **고등학교 수학 I의 지수함수 성질**만으로 이 감소 곡선 식을 완벽하게 유추할 수 있습니다.
        
        **📉 규칙적인 비율로 감소하는 현상 (등비수열과 지수함수)**
        * 만약 매 시간마다 체내에 남아있는 약물의 일정 비율(예: $30\%$)이 몸 밖으로 빠져나간다고 가정해 봅시다. 그럼 몸에 남는 양은 늘 전 시간의 $70\%(0.7)$가 됩니다.
        * **0시간 뒤 (처음):** $C_0$
        * **1시간 뒤:** $C_0 \times 0.7$
        * **2시간 뒤:** $C_0 \times 0.7 \times 0.7 = C_0 \times 0.7^2$
        * **$t$시간 뒤:** $C_0 \times 0.7^t$
        
        이처럼 **"시간이 흐름에 따라 남은 양의 '일정한 비율'이 지속적으로 감소하는 현상"**은 수학적으로 무조건 **지수함수($y = a \cdot b^x$, 단 $0 < b < 1$)** 모델이 될 수밖에 없습니다. 
        자연과학에서는 이 밑($b$)을 계산이 편리한 자연상수 $e$를 활용해 $e^{-k}$ 형태로 표현할 뿐이랍니다.
        """)

# =============================================================================
# TAB 2: 3단계 (역함수 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 공식 도출 및 복용 간격 예측</div>', unsafe_allow_html=True)
    
    # 3단계용 이상적 파라미터 제시상태
    opt_c0 = 25.0
    opt_k = 0.35
    true_half_life = np.log(2) / opt_k # 약 1.98시간
    
    st.markdown("""
    2단계 탐구를 통해 우리는 가장 적합한 모델이 **지수함수 모델**임을 알아냈습니다. 
    이제 분석을 통해 도출된 환자의 파라미터를 기반으로 맞춤형 복용 주기를 설계해 봅시다.
    """)
    
    # 💡 [이유 명시] 왜 반감기를 계산해야 하는가?
    st.markdown("### 💡 우리가 '반감기'를 직접 계산해야 하는 이유")
    st.markdown("""
    <div class="info-box">
        소실 속도 상수 $k = 0.35$라는 숫자는 수학적으로는 정밀하지만, 환자나 의사에게 <strong>"그래서 이 약을 몇 시간마다 먹어야 하나요?"</strong>라는 질문에 직관적인 답을 주지 못합니다.<br>
        반면 <strong>반감기($t_{1/2}$)</strong>는 약의 농도가 정확히 절반으로 줄어드는 <strong>'시간(hour) 단위'</strong>의 직관적인 지표를 제공합니다. 따라서 안전한 복용 주기를 설계하기 위해 $k$로부터 반감기를 환산하는 과정이 반드시 필요합니다.
    </div>
    """, unsafe_allow_html=True)
    
    # 2구획 레이아웃 (좌: 수식 및 학생 계산 미션 / 우: 역함수 원리)
    col_quiz, col_math = st.columns([1.2, 1])
    
    with col_quiz:
        st.markdown('<div class="quiz-box">', unsafe_allow_html=True)
        st.markdown("#### ✍️ [미션] 환자의 반감기($t_{1/2}$)를 직접 계산해 보세요!")
        st.write(f"현재 환자의 초기 농도는 **$C_0 = {opt_c0}$ mg/L**, 소실 상수는 **$k = {opt_k}$** 입니다.")
        st.latex(r"t_{1/2} = \frac{\ln 2}{k} \quad (\text{힌트: } \ln 2 \approx 0.693)")
        
        # 학생들이 직접 계산하여 입력하는 칸
        student_input = st.number_input("계산한 반감기 값을 입력하세요 (소수점 첫째 자리까지 권장):", min_value=0.0, max_value=12.0, step=0.1)
        
        # 정답 체크 로직 (어느 정도 오차 허용)
        is_correct = False
        if student_input > 0:
            if abs(student_input - true_half_life) < 0.15:
                st.success(f"🎉 정답입니다! 이 환자의 약물 반감기는 약 **{student_input:.1f}시간** 입니다. 아래에서 시뮬레이션 기능이 활성화되었습니다.")
                is_correct = True
            else:
                st.error("다시 한 번 계산해 보세요! 힌트: 0.693을 소실 상수 0.35로 나누어 보세요.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_math:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 역함수(로그)의 적용 원리")
        st.write("특정 목표 농도($C_{min}$)에 도달하는 정확한 시간($t$)을 구하기 위해, 우리는 지수함수의 역함수 관계인 **자연로그** 공식을 사용합니다.")
        st.latex(r"C = C_0 \cdot e^{-kt}")
        st.latex(r"\Rightarrow \ln\left(\frac{C}{C_0}\right) = -kt")
        st.latex(r"\Rightarrow t = \frac{-\ln(C/C_0)}{k}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 💡 정답을 맞추어야만 하단 복용 간격 예측 슬라이더와 그래프가 열리도록 락(Lock) 제어
    if is_correct:
        st.markdown("---")
        st.markdown("### 🎯 최소 유효 농도 설정 및 복용 주기 예측")
        st.write("약효가 완전히 사라지기 전(통증이 다시 시작되기 전), 약을 재복용해야 하는 기준선인 **최소 유효 농도**를 설정해 보세요.")
        
        target_c = st.slider("최소 유효 농도 ($C_{min}$ 단위: mg/L)", 1.0, 10.0, 5.0, 0.5)
        
        # 역함수 공식을 이용한 시간 예측 계산
        pred_time = -np.log(target_c / opt_c0) / opt_k
        
        st.markdown(f"### ⏱️ 예측 결과: 다음 약 복용 시점은 복용 후 **약 {pred_time:.1f}시간** 뒤입니다.")
        
        # Plotly 시뮬레이션 시각화
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
