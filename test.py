import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 페이지 기본 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(page_title="약물 대사 시뮬레이터", page_icon="💊", layout="wide")

# 학생 친화적 CSS 스타일 적용 (폰트 및 여백)
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; }
    .sub-title { font-size: 1.5rem; color: #047857; font-weight: 600; }
    .mission-box { background-color: #F0FDF4; padding: 20px; border-radius: 15px; border-left: 5px solid #10B981; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💊 앗! 몸속 약물은 다 어디로 갔을까?</div>', unsafe_allow_html=True)
st.write("고등학교 융합 프로젝트: 수집된 데이터를 통해 몸속 약물의 감소 규칙을 수학적으로 찾아내고 예측해 보세요!")

# -----------------------------------------------------------------------------
# [공통 데이터 생성] 2차시 탐구용 가상 환자 데이터셋
# -----------------------------------------------------------------------------
np.random.seed(42)
time_data = np.linspace(0, 12, 60) # 0시간부터 12시간까지 60개 시점
C_0_true = 25.0
k_true = 0.35
noise = np.random.normal(0, 0.8, len(time_data))
conc_data = C_0_true * np.exp(-k_true * time_data) + noise
conc_data = np.maximum(conc_data, 0.1) # 음수 방지
df_sample = pd.DataFrame({"Time_hours": time_data, "Concentration": conc_data})

# -----------------------------------------------------------------------------
# 사이드바 설정 (3차시용)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # 귀여운 약 아이콘
    st.header("⚙️ 시뮬레이션 조종기")
    st.info("이 조종기는 **3차시 탭**에서 약물 복용을 설계할 때 사용합니다. 자유롭게 조절해 보세요!")
    input_c0 = st.slider("🧪 약물 투여 직후 농도 (C₀)", 10.0, 50.0, 25.0)
    input_half_life = st.slider("⏳ 환자의 약물 반감기 (시간)", 1.0, 6.0, 2.0)
    calculated_k = np.log(2) / input_half_life

# -----------------------------------------------------------------------------
# 메인 화면 - 탭 구성
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🕵️‍♂️ 2차시: 사라진 약물의 수학적 비밀 찾기 (미션)", 
    "🩺 3차시: AI 의사가 되어 타이레놀 처방하기"
])

# -----------------------------------------------------------------------------
# 2차시 탭: 학생 주도적 함수식 탐구 미션
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="sub-title">미션: 데이터의 흐름을 지배하는 수식을 찾아라!</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mission-box">
        <strong>📝 탐구 활동 가이드</strong><br>
        1. 환자의 피를 뽑아 측정한 붉은색 점(데이터)들의 흐름을 관찰하세요.<br>
        2. 이 점들은 어떤 함수의 모양을 하고 있나요? 직선일까요, 곡선일까요?<br>
        3. 슬라이더를 움직여 파란색 곡선을 붉은색 점들에 최대한 포개어 보세요!
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ 수식 조립기")
        st.write("우리 배운 '지수함수' 공식을 이용해 곡선을 구부려 봅시다.")
        st.latex(r"C(t) = A \cdot e^{-b \cdot t}")
        
        st.write("⬇️ 아래 변수들을 조절해 일치율 95점 이상을 달성하세요!")
        manual_a = st.slider("초기 시작 높이 (A)", 10.0, 40.0, 15.0, step=0.5, help="약물을 처음 투여했을 때의 농도입니다.")
        manual_b = st.slider("감소하는 속도 (b)", 0.01, 1.00, 0.10, step=0.01, help="약물이 몸에서 빠져나가는 속도 상수입니다.")
        
        # 일치율(Score) 계산: R-squared 기반의 게임화 점수
        user_curve = manual_a * np.exp(-manual_b * time_data)
        ss_res = np.sum((conc_data - user_curve)**2)
        ss_tot = np.sum((conc_data - np.mean(conc_data))**2)
        r2 = 1 - (ss_res / ss_tot)
        score = max(0, int(r2 * 100))
        
        st.write("---")
        st.subheader("🎯 현재 일치율 점수")
        
        # 점수에 따른 피드백 이모지 및 색상 변화
        if score < 50:
            st.error(f"점수: {score}점 😭 (곡선이 점들과 너무 멀어요!)")
        elif score < 85:
            st.warning(f"점수: {score}점 😐 (조금 더 조절해 보세요!)")
        elif score < 95:
            st.info(f"점수: {score}점 😊 (거의 다 왔어요! 미세 조정을 해보세요!)")
        else:
            st.success(f"점수: {score}점 🏆 (완벽합니다! 수학자이신가요?)")

    with col2:
        # 그래프 시각화
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(time_data, conc_data, color="#EF4444", alpha=0.8, label="환자 혈중 농도 측정값 (데이터)")
        ax.plot(time_data, user_curve, color="#3B82F6", linewidth=3, 
                 label=f"내가 만든 수식: $C(t) = {manual_a:.1f} \\cdot e^{{-{manual_b:.2f}t}}$")
        ax.set_xlabel("약물 투여 후 경과 시간 (시간)", fontsize=11)
        ax.set_ylabel("혈중 약물 농도 (ug/mL)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", fontsize=10)
        
        # 점수 배경색 표시 
        ax.set_facecolor('#F8FAFC')
        st.pyplot(fig)

    # 95점 이상일 때만 열리는 비밀 해설 코너
    if score >= 95:
        st.balloons() # 축하 효과!
        st.markdown("---")
        st.markdown('<div class="sub-title">🎉 미션 성공! 감춰진 과학적 비밀 확인하기</div>', unsafe_allow_html=True)
        st.success("""
        여러분은 방금 컴퓨터(AI)가 최적의 선을 찾는 방법인 **최적화 알고리즘**을 마우스로 직접 수행해 낸 것입니다!
        
        **👩‍🔬 우리가 찾은 함수식의 의미:**
        * 이 곡선의 정체는 바로 **지수함수 (Exponential Decay)** 입니다.
        * 화학과 생명과학에서, 약물이 몸 밖으로 빠져나가는 속도는 '현재 내 몸에 남은 약의 양'에 비례합니다. 이를 미분방정식으로 쓰면 $\\frac{dC}{dt} = -kC$ 가 되는데, 이 식을 수학적으로 풀면 여러분이 방금 만든 $C(t) = C_0 \\cdot e^{-kt}$ 라는 예쁜 지수함수 공식이 탄생한답니다.
        * 이 원리를 바탕으로 3차시 탭으로 넘어가 진짜 의사처럼 타이레놀 처방을 내려봅시다!
        """)

# -----------------------------------------------------------------------------
# 3차시 탭: 타이레놀 효과 및 투약 시간 예측 (기존 코드 유지 및 디자인 개선)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="sub-title">🩺 3차시: 내 몸을 지키는 타이레놀 복용 타이밍!</div>', unsafe_allow_html=True)
    
    st.markdown("""
    왼쪽의 **[시뮬레이션 조종기]**를 조작하여 환자에게 딱 맞는 약효 지속 시간을 계산해 보세요.
    * 🟢 **치료 유효 영역 (안전존)**: 약 농도가 5 ~ 25 $\mu g/mL$ 사이에 있어야 약효가 안전하게 발휘됩니다.
    """)
    
    t_sim = np.linspace(0, 24, 300)
    c_sim = input_c0 * np.exp(-calculated_k * t_sim)
    
    if input_c0 > 5.0:
        duration_hours = np.log(input_c0 / 5.0) / calculated_k
    else:
        duration_hours = 0.0
        
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(t_sim, c_sim, color="#0D9488", linewidth=3, label="예측된 몸속 타이레놀 농도")
    ax3.axhline(y=5.0, color="#F59E0B", linestyle="--", linewidth=2, label="최소 유효 농도 (아픔이 사라짐)")
    ax3.axhline(y=25.0, color="#EF4444", linestyle=":", linewidth=2, label="최소 독성 농도 (간이 위험함!)")
    ax3.fill_between(t_sim, 0, c_sim, where=(c_sim >= 5.0)&(c_sim <= 25.0), color="#10B981", alpha=0.15, label="안전한 약효 발휘 구간")
    
    ax3.set_xlabel("시간 (hours)")
    ax3.set_ylabel("타이레놀 혈중 농도 (ug/mL)")
    ax3.set_ylim(0, max(input_c0 + 5, 35))
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")
    ax3.set_facecolor('#F8FAFC')
    st.pyplot(fig3)
    
    st.markdown("### 🏥 AI 주치의의 복용 가이드")
    if input_c0 >= 25.0:
        st.error(f"🚨 **위험해요!** 처음 먹은 양이 너무 많아 독성 농도(25 $\mu g/mL$)를 넘었습니다. 간이 다칠 수 있으니 복용량을 줄여주세요!")
    elif duration_hours > 0:
        st.success(f"✨ **완벽한 처방입니다.** 약효가 **{duration_hours:.1f}시간** 동안 부작용 없이 유지됩니다. 다음 알약은 약 {max(4.0, np.round(duration_hours)):.0f}시간 뒤에 드시는 것이 좋습니다.")
    else:
        st.warning("⚠️ 약을 너무 적게 먹어서 효과가 없습니다. (최소 유효 농도 5 미만)")
        
    with st.expander("🤔 깊이 생각해보기: 우리의 수학적 계산이 현실과 다른 점은 무엇일까요?"):
        st.markdown("""
        * 알약을 꿀꺽 삼키면 위장으로 들어가 소화되는데 시간이 걸리죠? 하지만 우리의 수학 모델(정맥 주사 모델)은 '먹자마자 1초 만에 핏속으로 100% 흡수되었다'고 가정했어요. 
        * 그래서 실제로는 농도가 천천히 올라갔다가 내려가는 '산 모양(⛰️)'의 그래프가 그려진답니다. 이를 수학적으로 풀려면 어떻게 해야 할까요? (힌트: 흡수되는 속도와 빠져나가는 속도를 동시에 계산하는 2계 미분방정식이 필요합니다!)
        """)
