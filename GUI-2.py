import streamlit as st
import numpy as np
import pandas as pd
import scipy as sci
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import legendre

def compute_jacobi_method(n):
    # Jacobi matrix method to calculate roots and weights
    k = np.arange(1., n)
    b = k / np.sqrt(4 * k * k - 1)
    A = np.diag(b, -1) + np.diag(b, 1)

    roots, eigenvectors = eigh(A)
    weights = 2 * (eigenvectors[0, :] ** 2)
    return roots, weights

def companion_matrix(coeffs):
    # Companion matrix function
    n = len(coeffs) - 1
    C = np.zeros((n, n))
    for i in range(n - 1):
        C[i + 1, i] = 1
    for i in range(n):
        C[i, -1] = -(coeffs[n - i] / coeffs[0])
    return C

def legendre_polynomial_derivative(coeffs):
    # Derivative of the Legendre polynomial
    return np.polyder(coeffs)

def compute_gauss_legendre_weights(n, coeffs, roots):
    # Function to compute Gauss-Legendre weights
    derivative_coeffs = legendre_polynomial_derivative(coeffs)
    derivative_values = np.polyval(derivative_coeffs, roots)
    weights = 2 / ((1 - roots**2) * derivative_values**2)
    return weights

# Streamlit UI
st.title("NUMERICAL METHODS ASSIGNMENT 2 (Gauss-Legendre Quadrature)")
st.markdown("""
This interactive tool allows you to explore **Gauss-Legendre Quadrature**, a numerical integration technique. 
Choose the number of points `n` to compute roots and weights using two different methods:
- **Jacobi Matrix Method**
- **Companion Matrix Method**

The comparison plot shows how results from these methods align, providing insights into the accuracy of both approaches.
""")

# Input for n
n = st.slider("Select value of n", min_value=2, max_value=64, value=30)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Jacobi Method", "Companion Matrix Method", "Comparison Plot"])

with tab1:
    st.header("Jacobi Matrix Method")
    st.markdown("""
    The **Jacobi Matrix Method** uses a tridiagonal matrix whose eigenvalues correspond to the roots of the 
    Legendre polynomial. This approach is computationally efficient and commonly used for Gauss-Legendre quadrature.
    """)
    if st.button("Calculate Jacobi Results"):
        roots_jacobi, weights_jacobi = compute_jacobi_method(n)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Roots")
            st.dataframe(pd.DataFrame(roots_jacobi, columns=['Root']))
        
        with col2:
            st.subheader("Weights")
            st.dataframe(pd.DataFrame(weights_jacobi, columns=['Weight']))

with tab2:
    st.header("Companion Matrix Method")
    st.markdown("""
    The **Companion Matrix Method** derives roots by constructing a companion matrix for the Legendre polynomial 
    of degree `n`. The eigenvalues of this matrix represent the roots, which can then be used to compute weights.
    """)
    if st.button("Calculate Companion Matrix Results"):
        P_n = legendre(n)
        coeffs = P_n.coefficients
        C = companion_matrix(coeffs)
        
        roots_companion, eigenvectors = np.linalg.eig(C)
        roots_companion = np.sort(roots_companion.real)
        weights_companion = compute_gauss_legendre_weights(n, coeffs, roots_companion)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Roots")
            st.dataframe(pd.DataFrame(roots_companion, columns=['Root']))
        
        with col2:
            st.subheader("Weights")
            st.dataframe(pd.DataFrame(weights_companion, columns=['Weight']))

with tab3:
    st.header("Comparison Plot")
    st.markdown("""
    The comparison plot below shows how the roots and weights from both methods align. This visual representation helps in understanding the accuracy and consistency of the results obtained from the Jacobi and Companion Matrix methods.
    """)
    if st.button("Generate Comparison Plot"):
        roots_jacobi, weights_jacobi = compute_jacobi_method(n)
        P_n = legendre(n)
        coeffs = P_n.coefficients
        C = companion_matrix(coeffs)
        roots_companion, eigenvectors = np.linalg.eig(C)
        roots_companion = np.sort(roots_companion.real)
        weights_companion = compute_gauss_legendre_weights(n, coeffs, roots_companion)

        fig, ax = plt.subplots()
        ax.plot(roots_jacobi, weights_jacobi, 'o', label="Jacobi Matrix Method")
        ax.plot(roots_companion, weights_companion, 'x', label="Companion Matrix Method")
        ax.set_xlabel("Roots")
        ax.set_ylabel("Weights")
        ax.set_title(f"Weights vs Roots for n = {n}")
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)
