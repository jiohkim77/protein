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


# 실험 3 수정된 코드 (app.py에서 해당 부분만 교체)

elif experiment == "🔢 실험 3: 헤시안 조건수":
    st.markdown('<div class="experiment-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔢 실험 3: 헤시안 행렬 조건수와 수렴성
    
    **목표**: 다변수 함수에서 조건수 κ = λ_max/λ_min과 최적화 난이도 관계 검증
    
    **함수**: f(x,y) = ax² + by² (서로 다른 곡률을 가진 2차원 그릇)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정 (더 간단하게)
    st.sidebar.markdown("---")
    st.sidebar.subheader("실험 3 설정")
    
    # 미리 정의된 조건수들 중 선택
    condition_options = {
        "쉬운 문제 (κ=1)": {"a": 1, "b": 1, "condition": 1},
        "보통 문제 (κ=4)": {"a": 4, "b": 1, "condition": 4}, 
        "어려운 문제 (κ=10)": {"a": 10, "b": 1, "condition": 10},
        "매우 어려운 문제 (κ=20)": {"a": 20, "b": 1, "condition": 20}
    }
    
    selected_problems = st.sidebar.multiselect(
        "문제 난이도 선택",
        list(condition_options.keys()),
        default=["쉬운 문제 (κ=1)", "보통 문제 (κ=4)", "어려운 문제 (κ=10)"]
    )
    
    start_x = st.sidebar.slider("시작점 x", 1.0, 3.0, 2.0)
    start_y = st.sidebar.slider("시작점 y", 1.0, 3.0, 2.0)
    max_lr = st.sidebar.slider("최대 학습률", 0.1, 0.5, 0.3)
    
    if st.sidebar.button("🚀 실험 3 실행", type="primary"):
        if not selected_problems:
            st.error("최소 하나의 문제를 선택해주세요!")
        else:
            with st.spinner("헤시안 조건수 분석 중..."):
                try:
                    # 선택된 문제들만 실험
                    selected_configs = [condition_options[prob] for prob in selected_problems]
                    
                    # 학습률 범위 (더 적게)
                    learning_rates = np.linspace(0.001, max_lr, 15)
                    all_results = []
                    
                    progress_bar = st.progress(0)
                    total_experiments = len(selected_configs) * len(learning_rates)
                    count = 0
                    
                    for config in selected_configs:
                        a, b_value = config["a"], config["b"]
                        condition_number = config["condition"]
                        theoretical_max_lr = 2 / max(2*a, 2*b_value)
                        
                        for lr in learning_rates:
                            # 2D 경사하강법: f(x,y) = ax² + by²
                            x, y = start_x, start_y
                            iterations = 0
                            max_iterations = 500  # 더 적게
                            tolerance = 1e-5
                            
                            # 발산 체크를 위한 초기값 저장
                            initial_distance = np.sqrt(x*x + y*y)
                            
                            for _ in range(max_iterations):
                                grad_x = 2 * a * x
                                grad_y = 2 * b_value * y
                                
                                gradient_norm = np.sqrt(grad_x**2 + grad_y**2)
                                
                                # 수렴 체크
                                if gradient_norm < tolerance:
                                    break
                                
                                # 파라미터 업데이트
                                x = x - lr * grad_x
                                y = y - lr * grad_y
                                iterations += 1
                                
                                # 발산 체크
                                current_distance = np.sqrt(x*x + y*y)
                                if current_distance > 10 * initial_distance:
                                    iterations = max_iterations  # 발산으로 처리
                                    break
                            
                            converged = iterations < max_iterations
                            final_distance = np.sqrt(x*x + y*y)
                            
                            all_results.append({
                                '문제': f"κ={condition_number}",
                                'a': a,
                                '조건수': condition_number,
                                '학습률': lr,
                                '반복수': iterations,
                                '수렴': converged,
                                '이론적_최대_학습률': theoretical_max_lr,
                                '최종_거리': final_distance
                            })
                            
                            count += 1
                            progress_bar.progress(count / total_experiments)
                    
                    if not selected_problems:
                        st.error("최소 하나의 문제를 선택해주세요!")
                        st.stop()  # ← return 대신 st.stop() 사용
                    df = pd.DataFrame(all_results)
                    
                    # 시각화 (간단하게 2x2)
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            '학습률 vs 수렴 반복수', 
                            '조건수 vs 최적 학습률', 
                            '수렴 성공률', 
                            '조건수별 평균 성능'
                        )
                    )
                    
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    
                    # 1. 학습률 vs 반복수 (조건수별)
                    for i, config in enumerate(selected_configs):
                        condition_num = config["condition"]
                        df_condition = df[df['조건수'] == condition_num]
                        
                        if len(df_condition) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_condition['학습률'], 
                                    y=df_condition['반복수'],
                                    mode='lines+markers', 
                                    name=f'κ={condition_num}',
                                    line=dict(color=colors[i % len(colors)]),
                                    marker=dict(size=4)
                                ),
                                row=1, col=1
                            )
                            
                            # 이론적 최대 학습률 표시
                            theoretical_max = df_condition['이론적_최대_학습률'].iloc[0]
                            if theoretical_max <= max_lr:
                                fig.add_vline(
                                    x=theoretical_max, 
                                    line_dash="dash", 
                                    line_color=colors[i % len(colors)], 
                                    row=1, col=1
                                )
                    
                    # 2. 조건수 vs 최적 학습률
                    optimal_results = []
                    for condition_num in df['조건수'].unique():
                        df_condition = df[df['조건수'] == condition_num]
                        converged_results = df_condition[df_condition['수렴'] == True]
                        
                        if len(converged_results) > 0:
                            # 가장 적은 반복수를 가진 학습률 찾기
                            optimal_idx = converged_results['반복수'].idxmin()
                            optimal_lr = converged_results.loc[optimal_idx, '학습률']
                            theoretical_max = converged_results['이론적_최대_학습률'].iloc[0]
                            
                            optimal_results.append({
                                '조건수': condition_num,
                                '최적_학습률': optimal_lr,
                                '이론적_최대': theoretical_max
                            })
                    
                    if optimal_results:
                        optimal_df = pd.DataFrame(optimal_results)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=optimal_df['조건수'], 
                                y=optimal_df['최적_학습률'],
                                mode='markers', 
                                name='실험적 최적 학습률',
                                marker=dict(size=12, color='blue', symbol='circle')
                            ),
                            row=1, col=2
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=optimal_df['조건수'], 
                                y=optimal_df['이론적_최대'],
                                mode='markers', 
                                name='이론적 최대 학습률',
                                marker=dict(size=12, color='red', symbol='x')
                            ),
                            row=1, col=2
                        )
                    
                    # 3. 수렴 성공률
                    convergence_rate = df.groupby('조건수')['수렴'].mean().reset_index()
                    convergence_rate.columns = ['조건수', '수렴률']
                    
                    fig.add_trace(
                        go.Bar(
                            x=convergence_rate['조건수'], 
                            y=convergence_rate['수렴률'],
                            name='수렴 성공률',
                            marker_color='lightgreen'
                        ),
                        row=2, col=1
                    )
                    
                    # 4. 조건수별 평균 성능
                    avg_performance = df[df['수렴'] == True].groupby('조건수')['반복수'].mean().reset_index()
                    if len(avg_performance) > 0:
                        fig.add_trace(
                            go.Bar(
                                x=avg_performance['조건수'], 
                                y=avg_performance['반복수'],
                                name='평균 반복수',
                                marker_color='lightblue'
                            ),
                            row=2, col=2
                        )
                    
                    # 레이아웃 업데이트
                    fig.update_layout(height=700, title_text="헤시안 조건수 실험 결과")
                    
                    # x, y축 라벨
                    fig.update_xaxes(title_text="학습률", row=1, col=1)
                    fig.update_yaxes(title_text="반복수", row=1, col=1)
                    fig.update_xaxes(title_text="조건수", row=1, col=2)
                    fig.update_yaxes(title_text="학습률", row=1, col=2)
                    fig.update_xaxes(title_text="조건수", row=2, col=1)
                    fig.update_yaxes(title_text="수렴률", row=2, col=1)
                    fig.update_xaxes(title_text="조건수", row=2, col=2)
                    fig.update_yaxes(title_text="평균 반복수", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 주요 결과 메트릭
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_convergence = df['수렴'].mean()
                        st.metric("📊 전체 수렴률", f"{total_convergence:.1%}")
                    
                    with col2:
                        if len(optimal_df) > 1:
                            # 조건수와 최적 학습률의 상관관계
                            correlation = np.corrcoef(
                                1/optimal_df['조건수'], 
                                optimal_df['최적_학습률']
                            )[0,1]
                            st.metric("🔗 상관계수 (1/κ vs α)", f"{correlation:.3f}")
                        else:
                            st.metric("🔗 상관계수", "계산 불가")
                    
                    with col3:
                        if len(optimal_df) > 0:
                            avg_error = np.mean(np.abs(
                                optimal_df['최적_학습률'] - optimal_df['이론적_최대']
                            ))
                            st.metric("📏 평균 오차", f"{avg_error:.4f}")
                        else:
                            st.metric("📏 평균 오차", "계산 불가")
                    
                    # 핵심 발견 요약
                    st.subheader("🔍 핵심 발견")
                    
                    if len(optimal_df) > 1:
                        # 조건수 효과 분석
                        min_condition = optimal_df['조건수'].min()
                        max_condition = optimal_df['조건수'].max()
                        min_lr = optimal_df[optimal_df['조건수'] == max_condition]['최적_학습률'].iloc[0]
                        max_lr = optimal_df[optimal_df['조건수'] == min_condition]['최적_학습률'].iloc[0]
                        
                        improvement_factor = max_lr / min_lr if min_lr > 0 else float('inf')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"""
                            **📈 조건수 효과**
                            - 조건수 {min_condition} → 최적 학습률: {max_lr:.3f}
                            - 조건수 {max_condition} → 최적 학습률: {min_lr:.3f}
                            - **학습률 차이**: {improvement_factor:.1f}배
                            """)
                        
                        with col2:
                            if correlation < -0.7:
                                st.success("✅ **이론 검증 성공!** 조건수가 클수록 최적 학습률이 작아짐")
                            elif correlation < -0.5:
                                st.info("✔️ **이론 부분 검증.** 조건수와 학습률의 역관계 확인")
                            else:
                                st.warning("⚠️ **추가 실험 필요.** 더 많은 조건수로 실험해보세요")
                    
                    # 상세 데이터
                    with st.expander("📊 상세 실험 데이터"):
                        # 요약 테이블
                        summary_data = []
                        for condition_num in sorted(df['조건수'].unique()):
                            df_cond = df[df['조건수'] == condition_num]
                            converged_df = df_cond[df_cond['수렴'] == True]
                            
                            if len(converged_df) > 0:
                                best_lr = converged_df.loc[converged_df['반복수'].idxmin(), '학습률']
                                min_iterations = converged_df['반복수'].min()
                                convergence_rate = len(converged_df) / len(df_cond)
                                theoretical_max = df_cond['이론적_최대_학습률'].iloc[0]
                            else:
                                best_lr = "수렴 실패"
                                min_iterations = "∞"
                                convergence_rate = 0
                                theoretical_max = df_cond['이론적_최대_학습률'].iloc[0]
                            
                            summary_data.append({
                                '조건수': condition_num,
                                '이론적_최대_학습률': f"{theoretical_max:.4f}",
                                '실험적_최적_학습률': best_lr if isinstance(best_lr, str) else f"{best_lr:.4f}",
                                '최소_반복수': min_iterations if isinstance(min_iterations, str) else int(min_iterations),
                                '수렴률': f"{convergence_rate:.1%}"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # 전체 원시 데이터
                        with st.expander("🔬 전체 원시 데이터"):
                            st.dataframe(df.round(4))
                    
                except Exception as e:
                    st.error(f"❌ 실험 중 오류 발생: {str(e)}")
                    st.info("💡 해결 방법: 더 적은 문제를 선택하거나 최대 학습률을 줄여보세요.")


# 실험 4 수정된 코드 (app.py에서 해당 부분만 교체)

elif experiment == "🧠 실험 4: 신경망 손실함수":
    st.markdown('<div class="experiment-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🧠 실험 4: 실제 신경망 손실함수 시뮬레이션
    
    **목표**: 실제 기계학습 문제(선형 회귀)에서 곡률 기반 이론 검증
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정 (더 작은 기본값)
    st.sidebar.markdown("---")
    st.sidebar.subheader("실험 4 설정")
    n_samples = st.sidebar.slider("데이터 샘플 수", 20, 100, 50)  # 기본값 50으로 줄임
    noise_level = st.sidebar.slider("노이즈 레벨", 0.0, 0.3, 0.1)   # 최대값 줄임
    n_epochs = st.sidebar.slider("에포크 수", 10, 50, 30)           # 기본값 30으로 줄임
    
    if st.sidebar.button("🚀 실험 4 실행", type="primary"):
        with st.spinner("신경망 학습 시뮬레이션 중..."):
            try:
                # 데이터 생성 (더 간단하게)
                np.random.seed(42)
                X = np.random.uniform(-2, 2, n_samples)
                y_true = 2 * X + 1
                noise = np.random.normal(0, noise_level, n_samples)
                y = y_true + noise
                
                # 학습률 설정 (더 적게)
                learning_rates = [0.01, 0.05, 0.1, 0.2]
                results = []
                
                progress_bar = st.progress(0)
                
                for i, lr in enumerate(learning_rates):
                    # 파라미터 초기화
                    w, b = 0.0, 0.0
                    
                    loss_history = []
                    curvature_history = []
                    w_history = []
                    
                    for epoch in range(n_epochs):
                        # Forward pass
                        y_pred = w * X + b
                        loss = np.mean((y_pred - y) ** 2)
                        
                        # Early stopping (중요!)
                        if loss < 1e-6:
                            break
                            
                        # Gradient computation
                        grad_w = np.mean(2 * (y_pred - y) * X)
                        grad_b = np.mean(2 * (y_pred - y))
                        
                        # Hessian approximation
                        hessian_ww = np.mean(2 * X ** 2)
                        hessian_bb = 2.0
                        
                        curvature = max(hessian_ww, hessian_bb)
                        
                        # 기록
                        loss_history.append(loss)
                        curvature_history.append(curvature)
                        w_history.append(w)
                        
                        # Parameter update
                        w -= lr * grad_w
                        b -= lr * grad_b
                        
                        # 발산 방지
                        if abs(w) > 10 or abs(b) > 10:
                            break
                    
                    results.append({
                        '학습률': lr,
                        '최종_손실': loss_history[-1] if loss_history else float('inf'),
                        '에포크': len(loss_history),
                        '최종_w': w,
                        '최종_b': b,
                        '평균_곡률': np.mean(curvature_history) if curvature_history else 0,
                        '손실_기록': loss_history,
                        '곡률_기록': curvature_history,
                        'w_기록': w_history
                    })
                    
                    progress_bar.progress((i + 1) / len(learning_rates))
                
                # 결과가 있는지 확인
                if not results or all(len(r['손실_기록']) == 0 for r in results):
                    st.error("❌ 실험이 제대로 실행되지 않았습니다. 설정값을 더 작게 해보세요!")
                    st.stop()    
                
                # 결과 분석
                valid_results = [r for r in results if len(r['손실_기록']) > 0]
                
                if not valid_results:
                    st.error("❌ 유효한 결과가 없습니다. 설정을 조정해주세요!")
                    st.stop()
                
                # 간단한 시각화
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('손실함수 수렴', '학습된 결과')
                )
                
                colors = ['blue', 'red', 'green', 'orange']
                
                # 1. 손실함수 수렴
                for i, result in enumerate(valid_results):
                    if len(result['손실_기록']) > 1:
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(result['손실_기록']))), 
                                y=result['손실_기록'],
                                mode='lines', 
                                name=f'LR={result["학습률"]}',
                                line=dict(color=colors[i % len(colors)])
                            ),
                            row=1, col=1
                        )
                
                # 2. 데이터와 결과
                fig.add_trace(
                    go.Scatter(
                        x=X, y=y, mode='markers', 
                        name='학습 데이터',
                        marker=dict(color='lightblue', size=6)
                    ),
                    row=1, col=2
                )
                
                # 참값 선
                x_line = np.linspace(X.min(), X.max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=2*x_line+1, mode='lines',
                        name='참값 (y=2x+1)', 
                        line=dict(color='green', width=3)
                    ),
                    row=1, col=2
                )
                
                # 최적 결과 선
                best_result = min(valid_results, key=lambda x: x['최종_손실'])
                y_pred_line = best_result['최종_w'] * x_line + best_result['최종_b']
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_pred_line, mode='lines',
                        name=f'학습 결과 (LR={best_result["학습률"]})',
                        line=dict(color='red', width=3, dash='dash')
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, title_text="실험 4 결과")
                fig.update_xaxes(title_text="에포크", row=1, col=1)
                fig.update_yaxes(title_text="손실", row=1, col=1)
                fig.update_xaxes(title_text="X", row=1, col=2)
                fig.update_yaxes(title_text="y", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 결과 메트릭
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 최적 학습률", f"{best_result['학습률']}")
                    st.metric("🔍 학습된 가중치 w", f"{best_result['최종_w']:.3f}")
                
                with col2:
                    st.metric("✅ 참값 가중치", "2.000")
                    st.metric("📊 가중치 오차", f"{abs(best_result['최종_w'] - 2):.3f}")
                
                with col3:
                    st.metric("🔄 학습된 편향 b", f"{best_result['최종_b']:.3f}")
                    st.metric("📈 최종 손실", f"{best_result['최종_손실']:.6f}")
                
                # 성공/실패 판단
                w_error = abs(best_result['최종_w'] - 2.0)
                if w_error < 0.1:
                    st.success(f"✅ 실험 성공! 가중치 오차가 {w_error:.3f}로 매우 정확합니다.")
                elif w_error < 0.5:
                    st.info(f"✔️ 실험 완료. 가중치 오차가 {w_error:.3f}로 양호합니다.")
                else:
                    st.warning(f"⚠️ 가중치 오차가 {w_error:.3f}입니다. 설정을 조정해보세요.")
                
                # 상세 결과
                with st.expander("📊 상세 실험 데이터"):
                    df_simple = pd.DataFrame([{
                        '학습률': r['학습률'],
                        '최종_손실': r['최종_손실'],
                        '에포크': r['에포크'],
                        '최종_w': r['최종_w'],
                        '최종_b': r['최종_b']
                    } for r in valid_results])
                    st.dataframe(df_simple.round(4))
                
            except Exception as e:
                st.error(f"❌ 실험 중 오류가 발생했습니다: {str(e)}")
                st.info("💡 해결 방법: 설정값을 더 작게 하거나 페이지를 새로고침해보세요.")
