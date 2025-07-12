import streamlit as st
import numpy as np
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Constants
L = 1.0
u_0 = 0
u_L = 1
h = 0.01
N = int(L / h)

def generate_y(N):
    return np.linspace(0, 1, N + 1)

def analytical_solution(P, y):
    return P * 0.5 * y * (1 - y) + y

def explicit_euler(P, v0, y):
    u = np.zeros(len(y))
    v = np.zeros(len(y))
    
    u[0] = u_0
    v[0] = v0
    
    for i in range(len(y) - 1):
        u[i + 1] = u[i] + h * v[i]
        v[i + 1] = v[i] - h * P
    
    return u

def implicit_euler(P, v0, y, tol=1e-6, itr=100):
    u = np.zeros(len(y))
    v = np.zeros(len(y))
    
    u[0] = u_0
    v[0] = v0
    
    for i in range(len(y) - 1):
        v_guess = v[i]
        for _ in range(itr):
            F = v_guess - v[i] + h * P
            F_prime = 1
            
            v_next = v_guess - F / F_prime
            if abs(v_next - v_guess) < tol:
                break
            v_guess = v_next
        
        v[i + 1] = v_next
        u[i + 1] = u[i] + h * v[i + 1]
    
    return u

def shooting_method(P, y, method, tol=1e-6, itr=100):
    v0_low, v0_high = -10, 10
    
    for _ in range(itr):
        v0 = (v0_low + v0_high) / 2
        u = method(P, v0, y)
        
        if abs(u[-1] - u_L) < tol:
            return u, v0
            
        if u[-1] > u_L:
            v0_high = v0
        else:
            v0_low = v0
    
    return None, None

def generate_jacobian():
    return np.array([[0, 1], [0, 0]])

def eigen_values():
    J = np.array([[0, 1], [0, 0]])
    return J,eigvals(J).real

def lu_decomposition(matrix):
    A = np.copy(matrix)
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i][i] = 1
        for k in range(i, n):
            U[i][k] = A[i][k]
            for j in range(i):
                U[i][k] -= L[i][j] * U[j][k]
        for k in range(i + 1, n):
            L[k][i] = A[k][i]
            for j in range(i):
                L[k][i] -= L[k][j] * U[j][i]
            L[k][i] /= U[i][i]
    
    return L, U

def lu_method(A, B, L, U):
    n = A.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    
    for i in range(n):
        y[i] = (B[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x

def bvp(P, h, N):
    y = generate_y(N)
    A = np.zeros((N - 1, N - 1))
    b = np.full(N - 1, -P * h**2)
    
    for i in range(N - 1):
        if i > 0:
            A[i, i - 1] = 1
        A[i, i] = -2
        if i < N - 2:
            A[i, i + 1] = 1
    
    b[0] -= u_0
    b[-1] -= u_L
    
    L, U = lu_decomposition(A)
    u_inner = lu_method(A, b, L, U)
    u = np.zeros(N + 1)
    u[0] = u_0
    u[1:N] = u_inner
    u[N] = u_L
    
    return u, y

def create_plotly_comparison(P_values, method_name, method_func=None):
    y = generate_y(N)
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['x', 'square', 'diamond', 'circle']
    
    for idx, P in enumerate(P_values):
        u_analytical = analytical_solution(P, y)
        fig.add_trace(
            go.Scatter(
                x=u_analytical,
                y=y,
                name=f'Analytical (P={P})',
                mode='lines+markers',
                marker=dict(
                    symbol=markers[idx % len(markers)],
                    size=8,
                    color=colors[idx % len(colors)]
                ),
                line=dict(color=colors[idx % len(colors)]),
                legendgroup=f'group{idx}'
            )
        )
        
        if method_name == "All Methods":
            u_explicit, _ = shooting_method(P, y, explicit_euler)
            if u_explicit is not None:
                fig.add_trace(
                    go.Scatter(
                        x=u_explicit,
                        y=y,
                        name=f'Explicit Euler (P={P})',
                        line=dict(color=colors[idx % len(colors)], dash='dot'),
                        legendgroup=f'group{idx}'
                    )
                )
            
            u_implicit, _ = shooting_method(P, y, implicit_euler)
            if u_implicit is not None:
                fig.add_trace(
                    go.Scatter(
                        x=u_implicit,
                        y=y,
                        name=f'Implicit Euler (P={P})',
                        line=dict(color=colors[idx % len(colors)], dash='dash'),
                        legendgroup=f'group{idx}'
                    )
                )
            
            u_bvp, y_bvp = bvp(P, 0.0625, 16)
            fig.add_trace(
                go.Scatter(
                    x=u_bvp,
                    y=y_bvp,
                    name=f'BVP (P={P})',
                    line=dict(color=colors[idx % len(colors)], dash='solid'),
                    legendgroup=f'group{idx}'
                )
            )
        elif method_name == "BVP":
            u_numerical, y_bvp = bvp(P, 0.0625, 16)
            fig.add_trace(
                go.Scatter(
                    x=u_numerical,
                    y=y_bvp,
                    name=f'BVP (P={P})',
                    line=dict(color=colors[idx % len(colors)]),
                    legendgroup=f'group{idx}'
                )
            )
        elif method_func is not None:
            u_numerical, _ = shooting_method(P, y, method_func)
            if u_numerical is not None:
                fig.add_trace(
                    go.Scatter(
                        x=u_numerical,
                        y=y,
                        name=f'{method_name} (P={P})',
                        line=dict(color=colors[idx % len(colors)]),
                        legendgroup=f'group{idx}'
                    )
                )
    
    fig.update_layout(
        title=dict(
            text=f'Comparison of {method_name} with Analytical Solution',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='u(y)',
        yaxis_title='y',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        template='plotly_white'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Numerical Methods Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color:  F0F8FF;
            color: white;
        }
        .stButton>button:hover {
            background-color:  F0F8FF;
        }
        .main > div {
            padding: 2rem;
            border-radius: 10px;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px;
            gap: 1px;
            padding: 10px;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("download.jpeg", width=150)
        st.title("Configuration")
        
        st.markdown("### P Values")
        P_values_input = st.text_input(
            "Enter P values",
            value="1.0, 2.0",
            help="Enter numbers separated by commas (e.g., 1.0, 2.0, 3.0)"
        )
        
        with st.expander("Advanced"):
            st.number_input("Step size (h)", value=0.01, step=0.001, format="%.3f")
            st.number_input("Tolerance", value=1e-6, format="%.0e")
            st.number_input("Max iterations", value=100, step=10)
        
        if st.button("Calculate Eigenvalues", type="secondary"):
            with st.spinner("Calculating..."):
                jacobian, eigenvalues = eigen_values()
                st.success("Calculation complete!")
                
                # Display Jacobian matrix
                st.write("Jacobian Matrix:")
                st.write(pd.DataFrame(jacobian))
                
                # Display eigenvalues
                st.write("Eigenvalues:")
                st.write(eigenvalues)
    
    st.title("Numerical Methods Assignment-3")
    st.markdown("""
    This interactive tool helps you analyze and compare different numerical methods 
    for solving boundary value problems. Choose a comparison method below and 
    adjust parameters in the sidebar to explore the results.
    """)
    
    try:
        P_values = [float(p.strip()) for p in P_values_input.split(",")]
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 All Methods",
            "🔵 Explicit Euler",
            "🔸 Implicit Euler",
            "📈 BVP"
        ])
        
        with tab1:
            st.header("Complete Method Comparison")
            if st.button("Generate Comparison", key="all"):
                with st.spinner("Generating plot..."):
                    fig = create_plotly_comparison(P_values, "All Methods")
                    st.plotly_chart(fig, use_container_width=True)
                st.success("Plot generated successfully!")
        
        with tab2:
            st.header("Explicit Euler Analysis")
            if st.button("Compare Explicit Euler", key="explicit"):
                with st.spinner("Analyzing..."):
                    fig = create_plotly_comparison(P_values, "Explicit Euler", explicit_euler)
                    st.plotly_chart(fig, use_container_width=True)
                st.success("Analysis complete!")
        
        with tab3:
            st.header("Implicit Euler Analysis")
            if st.button("Compare Implicit Euler", key="implicit"):
                with st.spinner("Analyzing..."):
                    fig = create_plotly_comparison(P_values, "Implicit Euler", implicit_euler)
                    st.plotly_chart(fig, use_container_width=True)
                st.success("Analysis complete!")
        
        with tab4:
            st.header("Boundary Value Problem Analysis")
            if st.button("Compare BVP Finite Difference Method", key="bvp"):
                with st.spinner("Analyzing..."):
                    fig = create_plotly_comparison(P_values, "BVP")
                    st.plotly_chart(fig, use_container_width=True)
                st.success("Analysis complete!")
    
    except ValueError:
        st.error("⚠️ Please enter valid numerical values separated by commas")
        st.info("Example: 1.0, 2.0, 3.0")

if __name__ == "__main__":
    main()
