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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">약물 농도 수학적 모델링 융합 탐구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">데이터를 관찰하고, 나만의 수학적 모델을 구축한 뒤, 이를 이용해 실제 복용 시간을 예측해 봅시다.</div>', unsafe_allow_html=True)

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
    st.write("아래 세 가지 함수 중 하나를 선택하고 슬라이더를 조절해 점들에 가장 잘 맞는 곡선을 직접 만들어 보세요. **그래프 아래쪽의 '잔차(오차)'가 특정한 패턴 없이 무작위로 흩어져야 완벽한 모델입니다!**")

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

    with st.expander("✅ [탐구 결론] 세 가지 함수를 모두 조작해 보았나요? 과학적 결론을 확인해 봅시다."):
        st.success("""
        **가장 완벽한 수학적 모델은 '지수 함수'입니다.**
        
        * 1차 함수(직선)나 2차 함수는 슬라이더를 아무리 잘 조절해 점수를 90점 이상으로 올려도, **아래쪽 잔차 그래프에 U자나 V자 모양의 뚜렷한 패턴**이 계속 남게 됩니다. 오직 **지수 함수만이 잔차를 0 주변에 무작위로 흩어지게** 만들 수 있습니다.
        """)

# =============================================================================
# TAB 2: 3단계 (수식 도출 및 복용 간격 예측)
# =============================================================================
with tab2:
    st.markdown('<div class="step-title">3단계: 수식 도출 및 맞춤형 복용 주기 예측</div>', unsafe_allow_html=True)
    
    # 이상적 파라미터
    opt_c0 = 25.0
    opt_k = 0.35
    true_half_life = np.log(2) / opt_k # 약 1.98시간
    
    # 💡 1. 미분방정식 없이 지수함수 추측 (직관적 탐구)
    st.markdown("### 🧠 1. 지수함수 모델의 직관적 이해 (미분방정식 없이)")
    with st.expander("👉 약물 감소의 수학적 비밀 알아보기 (클릭)", expanded=True):
        st.markdown("""
        어려운 미분방정식을 모르더라도, 우리는 **고등학교 수학 I의 등비수열** 원리로 이 식을 유추할 수 있습니다.
        
        약물이 체내에서 빠져나갈 때는 매 시간 **'남아있는 양의 일정한 비율'**만큼 빠져나갑니다. (예를 들어, 매시간 남은 양의 30%씩 감소한다면, 몸에는 전 시간의 **70%(0.7)**가 남게 됩니다.)
        * **0시간 (처음):** $C_0$
        * **1시간 뒤:** $C_0 \times 0.7$
        * **2시간 뒤:** $C_0 \times 0.7 \times 0.7 = C_0 \times 0.7^2$
        * **$t$시간 뒤:** $C_0 \times 0.7^t$
        
        이처럼 일정한 비율로 감소하는 현상은 수학적으로 무조건 **지수함수($y = a \cdot b^x$)** 형태가 됩니다. 과학에서는 이 밑($b$)을 계산하기 편한 자연상수 $e$를 활용하여 $e^{-k}$ 로 표현할 뿐이랍니다! 즉, 모델은 **$C(t) = C_0 \cdot e^{-kt}$** 가 됩니다.
        """)

    st.markdown("---")

    # 💡 2. 반감기 스스로 계산하기
    st.markdown("### ✍️ [미션 1] 약물 반감기($t_{1/2}$) 직접 계산하기")
    col_m1, col_m1_hint = st.columns([1, 1])
    
    with col_m1:
        st.markdown('<div class="mission-box">', unsafe_allow_html=True)
        st.write(f"현재 환자의 초기 농도는 **$C_0 = {opt_c0}$**, 소실 속도 상수는 **$k = {opt_k}$** 입니다.")
        st.write("반감기는 약물 농도가 처음의 정확히 절반($C_0 / 2$)으로 줄어드는 시간입니다. 환자의 반감기를 계산해 보세요.")
        student_half_life = st.number_input("계산한 반감기를 입력하세요 (소수 첫째 자리 권장):", min_value=0.0, max_value=10.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_m1_hint:
        st.info("""
        **💡 반감기 계산 힌트**
        1. 지수함수 식에 반감기 조건을 대입합니다: $\\frac{1}{2} C_0 = C_0 \cdot e^{-0.35t}$
        2. 양변을 $C_0$로 나눕니다: $\\frac{1}{2} = e^{-0.35t}$
        3. 양변에 자연로그($\\ln$)를 취하여 $t$값을 구해보세요. (단, $\\ln 2 \approx 0.693$)
        """)

    # 미션 1 통과 여부 확인
    is_m1_correct = False
    if student_half_life > 0:
        if abs(student_half_life - true_half_life) < 0.15: # 1.9 ~ 2.1 사이 정답 인정
            st.success(f"🎉 훌륭합니다! 환자의 반감기는 약 **{student_half_life:.1f}시간** 입니다. 다음 미션이 열렸습니다.")
            is_m1_correct = True
        else:
            st.error("아쉽네요! 다시 한 번 계산해 보세요. (힌트: 0.693 ÷ 0.35)")

    # 💡 3. 역함수 스스로 도출하기 (미션 1 성공 시 오픈)
    if is_m1_correct:
        st.markdown("---")
        st.markdown("### 🔄 [미션 2] 목표 농도 도달 시간 구하기 (역함수 찾기)")
        st.write("안전한 복용 주기를 알려면, 약물이 특정 목표 농도($C$)로 떨어지는 데 걸리는 시간($t$)을 알아야 합니다.")
        st.write("지수함수 $C = C_0 \cdot e^{-kt}$ 를 **$t$에 관해 정리(역함수)**하면 다음 중 어떤 식이 될까요?")
        
        formula_choice = st.radio("올바른 역함수 수식을 선택하세요:", 
                                  ["선택하세요", 
                                   "1. $t = \\frac{C_0 - C}{k}$", 
                                   "2. $t = \\frac{-\\ln(C / C_0)}{k}$", 
                                   "3. $t = -k \cdot \\ln(C / C_0)$"])
        
        is_m2_correct = False
        if formula_choice == "2. $t = \\frac{-\\ln(C / C_0)}{k}$":
            st.success("🎉 완벽합니다! 양변을 $C_0$로 나누고 자연로그($\\ln$)를 취하여 로그함수를 정확히 도출해 냈습니다. 최종 시뮬레이션이 활성화되었습니다.")
            is_m2_correct = True
        elif formula_choice != "선택하세요":
            st.error("오답입니다. 식을 풀 때 밑이 $e$인 지수를 없애기 위해 어떤 로그를 취해야 할지 생각해 보세요.")

        # 💡 4. 최소 유효 농도 설정 및 예측 (미션 2 성공 시 오픈)
        if is_m2_correct:
            st.markdown("---")
            st.markdown("### 🎯 [최종 단계] 최소 유효 농도 설정 및 복용 시점 예측")
            st.write("우리가 직접 도출한 역함수 공식을 바탕으로, 환자의 약효가 떨어지기 전 다음 약을 먹어야 할 시점을 예측해 봅시다.")
            
            target_c = st.slider("최소 유효 농도 설정 ($C_{min}$ 단위: mg/L)", 1.0, 10.0, 5.0, 0.5)
            
            # 학생이 직접 고른 역함수 공식을 활용해 컴퓨터가 시간 예측
            pred_time = -np.log(target_c / opt_c0) / opt_k
            
            st.markdown(f"### 💡 예측 결과: 다음 약 복용 시점은 투여 후 약 **{pred_time:.1f}시간** 뒤입니다.")
            
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
