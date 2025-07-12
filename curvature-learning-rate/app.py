import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# 페이지 설정
st.set_page_config(
    page_title="곡률-학습속도 이론 검증",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐싱을 위한 함수들
@st.cache_data
def generate_sample_data(n_samples=100, noise_level=0.2, seed=42):
    """실험 4용 샘플 데이터 생성"""
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, n_samples)
    y_true = 2 * X + 1
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    return X, y, y_true

@st.cache_data
def run_gradient_descent(func_type, start_point, learning_rate, max_iterations=1000):
    """경사하강법 실행 (캐싱됨)"""
    
    # 함수 정의
    if func_type == "quadratic":
        func = lambda x: x**2
        grad_func = lambda x: 2*x
        second_deriv = lambda x: 2
    elif func_type == "scaled_quadratic":
        func = lambda x: 10*x**2
        grad_func = lambda x: 20*x
        second_deriv = lambda x: 20
    else:  # quartic
        func = lambda x: x**2 + 0.1*x**4
        grad_func = lambda x: 2*x + 0.4*x**3
        second_deriv = lambda x: 2 + 1.2*x**2
    
    x = start_point
    path = [x]
    tolerance = 1e-6
    
    for i in range(max_iterations):
        gradient = grad_func(x)
        if abs(gradient) < tolerance:
            break
        x = x - learning_rate * gradient
        path.append(x)
    
    return np.array(path), i + 1, second_deriv(start_point)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .experiment-box {
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .theory-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<h1 class="main-header">📊 곡률-학습속도 이론의 실험적 검증</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="theory-box">
<h3>🧮 핵심 이론</h3>
<p><strong>최적 학습률</strong>: α* = 1/f''(x)</p>
<p><strong>수렴 조건</strong>: α < 2/λ_max</p>
<p><strong>조건수</strong>: κ = λ_max/λ_min</p>
</div>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.title("🔬 실험 제어판")
experiment = st.sidebar.selectbox(
    "실험 선택",
    ["🎯 실험 1: 최적 학습률 검증", 
     "🛤️ 실험 2: 경사하강법 경로", 
     "🔢 실험 3: 헤시안 조건수", 
     "🧠 실험 4: 신경망 손실함수"]
)

# 실험 1: 최적 학습률 검증
if experiment == "🎯 실험 1: 최적 학습률 검증":
    st.markdown('<div class="experiment-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 실험 1: 이론적 최적 학습률 검증
    
    **목표**: 다양한 곡률 함수에서 이론적 최적 학습률 α* = 1/f''(x) 검증
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.markdown("---")
    st.sidebar.subheader("실험 1 설정")
    func_choice = st.sidebar.selectbox("함수 선택", 
        ["f(x) = x²", "f(x) = 10x²", "f(x) = x² + 0.1x⁴"])
    start_point = st.sidebar.slider("시작점", -3.0, 3.0, 2.0)
    
    # 함수 타입 매핑
    func_map = {
        "f(x) = x²": "quadratic",
        "f(x) = 10x²": "scaled_quadratic", 
        "f(x) = x² + 0.1x⁴": "quartic"
    }
    
    if st.sidebar.button("🚀 실험 1 실행", type="primary"):
        with st.spinner("실험 진행 중..."):
            # 학습률 범위 설정
            learning_rates = np.linspace(0.01, 1.0, 20)
            results = []
            
            # 프로그레스 바
            progress_bar = st.progress(0)
            
            for i, lr in enumerate(learning_rates):
                path, iterations, curvature = run_gradient_descent(
                    func_map[func_choice], start_point, lr
                )
                
                theoretical_lr = 1 / curvature
                converged = iterations < 1000
                
                results.append({
                    '학습률': lr,
                    '반복수': iterations if converged else 1000,
                    '수렴': converged,
                    '이론적_최적': theoretical_lr,
                    '곡률': curvature
                })
                
                progress_bar.progress((i + 1) / len(learning_rates))
            
            # 결과 분석
            df = pd.DataFrame(results)
            optimal_idx = df['반복수'].idxmin()
            experimental_optimal = df.loc[optimal_idx, '학습률']
            theoretical_optimal = df.loc[0, '이론적_최적']
            
            # 시각화
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['학습률'], 
                    y=df['반복수'],
                    mode='lines+markers',
                    name='수렴 반복수',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # 최적점 표시
                fig.add_trace(go.Scatter(
                    x=[experimental_optimal],
                    y=[df.loc[optimal_idx, '반복수']],
                    mode='markers',
                    name='실험적 최적점',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                # 이론적 최적 학습률 선
                fig.add_vline(
                    x=theoretical_optimal,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"이론적 최적: {theoretical_optimal:.3f}"
                )
                
                fig.update_layout(
                    title="학습률 vs 수렴 반복수",
                    xaxis_title="학습률",
                    yaxis_title="반복수",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🎯 실험적 최적 학습률", f"{experimental_optimal:.4f}")
                st.metric("📐 이론적 최적 학습률", f"{theoretical_optimal:.4f}")
                error = abs(experimental_optimal - theoretical_optimal)
                st.metric("📊 절대 오차", f"{error:.4f}")
                st.metric("📈 곡률 f''(x)", f"{df.loc[0, '곡률']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 결과 분석
            error_percentage = error / theoretical_optimal * 100
            if error_percentage < 5:
                st.success(f"✅ 이론과 실험이 매우 잘 일치합니다! (오차: {error_percentage:.1f}%)")
            elif error_percentage < 15:
                st.info(f"✔️ 이론과 실험이 잘 일치합니다. (오차: {error_percentage:.1f}%)")
            else:
                st.warning(f"⚠️ 이론과 실험에 차이가 있습니다. (오차: {error_percentage:.1f}%)")
            
            # 상세 데이터
            with st.expander("📊 상세 실험 데이터"):
                st.dataframe(df.round(4))

# 실험 2: 경사하강법 경로
elif experiment == "🛤️ 실험 2: 경사하강법 경로":
    st.markdown('<div class="experiment-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🛤️ 실험 2: 경사하강법 경로와 곡률 변화
    
    **목표**: 곡률이 변하는 함수에서 학습 과정 시각화 및 적응적 학습률 필요성 확인
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.markdown("---")
    st.sidebar.subheader("실험 2 설정")
    start_point = st.sidebar.slider("시작점", -3.0, 3.0, 2.5)
    selected_lrs = st.sidebar.multiselect(
        "학습률 선택", 
        [0.05, 0.1, 0.2, 0.3, 0.5, 0.8], 
        default=[0.1, 0.3, 0.5]
    )
    max_iter = st.sidebar.slider("최대 반복수", 20, 100, 50)
    
    if st.sidebar.button("🚀 실험 2 실행", type="primary"):
        if not selected_lrs:
            st.error("최소 하나의 학습률을 선택해주세요!")
        else:
            with st.spinner("경로 추적 중..."):
                # f(x) = x² + 0.1x⁴ 함수 사용
                func = lambda x: x**2 + 0.1*x**4
                grad_func = lambda x: 2*x + 0.4*x**3
                second_deriv = lambda x: 2 + 1.2*x**2
                
                # 함수 시각화용 데이터
                x_range = np.linspace(-3, 3, 1000)
                y_range = [func(x) for x in x_range]
                
                # 서브플롯 생성
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('함수와 최적화 경로', '파라미터 변화', '곡률 변화', '이론적 최적 학습률'),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                # 함수 그래프
                fig.add_trace(
                    go.Scatter(x=x_range, y=y_range, name='f(x) = x² + 0.1x⁴', 
                              line=dict(color='black', width=2)),
                    row=1, col=1
                )
                
                colors = px.colors.qualitative.Set1
                
                for i, lr in enumerate(selected_lrs):
                    # 경사하강법 실행
                    x = start_point
                    path = [x]
                    
                    for _ in range(max_iter):
                        gradient = grad_func(x)
                        if abs(gradient) < 1e-6:
                            break
                        x = x - lr * gradient
                        path.append(x)
                    
                    path = np.array(path)
                    path_y = [func(x) for x in path]
                    curvatures = [second_deriv(x) for x in path]
                    theoretical_lrs = [1/second_deriv(x) for x in path]
                    
                    color = colors[i % len(colors)]
                    
                    # 1. 함수와 경로
                    fig.add_trace(
                        go.Scatter(x=path, y=path_y, mode='lines+markers',
                                  name=f'LR={lr}', line=dict(color=color, width=3),
                                  marker=dict(size=4)),
                        row=1, col=1
                    )
                    
                    # 2. 파라미터 변화
                    fig.add_trace(
                        go.Scatter(x=list(range(len(path))), y=path, mode='lines',
                                  name=f'위치 (LR={lr})', line=dict(color=color),
                                  showlegend=False),
                        row=1, col=2
                    )
                    
                    # 3. 곡률 변화
                    fig.add_trace(
                        go.Scatter(x=list(range(len(curvatures))), y=curvatures, mode='lines',
                                  name=f'곡률 (LR={lr})', line=dict(color=color),
                                  showlegend=False),
                        row=2, col=1
                    )
                    
                    # 4. 이론적 최적 학습률
                    fig.add_trace(
                        go.Scatter(x=list(range(len(theoretical_lrs))), y=theoretical_lrs, mode='lines',
                                  name=f'최적 LR (LR={lr})', line=dict(color=color),
                                  showlegend=False),
                        row=2, col=2
                    )
                
                fig.update_layout(height=700, title_text="경사하강법 경로 분석")
                fig.update_xaxes(title_text="x", row=1, col=1)
                fig.update_yaxes(title_text="f(x)", row=1, col=1)
                fig.update_xaxes(title_text="반복", row=1, col=2)
                fig.update_yaxes(title_text="파라미터 위치", row=1, col=2)
                fig.update_xaxes(title_text="반복", row=2, col=1)
                fig.update_yaxes(title_text="곡률", row=2, col=1)
                fig.update_xaxes(title_text="반복", row=2, col=2)
                fig.update_yaxes(title_text="최적 학습률", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 분석 결과
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("시작점 곡률", f"{second_deriv(start_point):.3f}")
                with col2:
                    st.metric("시작점 이론적 최적 LR", f"{1/second_deriv(start_point):.3f}")
                with col3:
                    final_curvature = second_deriv(0)  # 최적점에서의 곡률
                    st.metric("최적점 곡률", f"{final_curvature:.3f}")

# 실험 3: 헤시안 조건수
elif experiment == "🔢 실험 3: 헤시안 조건수":
    st.markdown('<div class="experiment-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔢 실험 3: 헤시안 행렬 조건수와 수렴성
    
    **목표**: 다변수 함수에서 조건수 κ = λ_max/λ_min과 최적화 난이도 관계 검증
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.markdown("---")
    st.sidebar.subheader("실험 3 설정")
    a_values = st.sidebar.multiselect("a 값 선택", [1, 2, 5, 10, 20], default=[1, 5, 10])
    b_value = st.sidebar.slider("b 값", 1, 3, 1)
    start_x = st.sidebar.slider("시작점 x", -3.0, 3.0, 2.0)
    start_y = st.sidebar.slider("시작점 y", -3.0, 3.0, 2.0)
    
    if st.sidebar.button("🚀 실험 3 실행", type="primary"):
        if not a_values:
            st.error("최소 하나의 a 값을 선택해주세요!")
        else:
            with st.spinner("헤시안 분석 중..."):
                learning_rates = np.linspace(0.001, 0.3, 25)
                all_results = []
                
                progress_bar = st.progress(0)
                total_experiments = len(a_values) * len(learning_rates)
                count = 0
                
                for a in a_values:
                    condition_number = max(2*a, 2*b_value) / min(2*a, 2*b_value)
                    theoretical_max_lr = 2 / max(2*a, 2*b_value)
                    
                    for lr in learning_rates:
                        # 2D 경사하강법: f(x,y) = ax² + by²
                        x, y = start_x, start_y
