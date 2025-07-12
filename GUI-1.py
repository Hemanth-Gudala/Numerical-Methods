import streamlit as st
import numpy as np
import pandas as pd
import scipy.linalg as sci
import sympy as sp
from numpy.linalg import eig

# Keep all original numerical functions
def lu_decomposition(matrix, y):
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
        if U[i, i] == 0:
            for g in range(n):
                A[g, g] += 1
            y[0] += 1
            return lu_decomposition(A, y)
        for k in range(i + 1, n):
            L[k][i] = A[k][i]
            for j in range(i):
                L[k][i] -= L[k][j] * U[j][i]
            L[k][i] /= U[i][i]

    return L, U

def eigen_values(matrix, iterations, tol):
    B = matrix.copy()
    n = B.shape[0]
    Bprev = np.eye(5)
    I = np.eye(5)
    eigenvalues = np.zeros(5)

    for _ in range(iterations):
        y = [0]
        L, U = lu_decomposition(B, y)
        B = np.dot(U, L)

        if np.max(np.abs(B-Bprev)) < tol and (np.max(np.abs(U-I))) < tol:
            eigenvalues = np.diag(L)
            break

        if np.max(np.abs(B-Bprev)) < tol and np.max(np.abs(L-I)) < tol:
            eigenvalues = np.diag(U)
            break

        Bprev = B

    return eigenvalues

def determinant(eigenv):
    eigenvalues = eigenv.copy()
    det = np.prod(eigenvalues)
    return det

def condition_number(n):
    max_eigenvalue = max(np.abs(n))
    min_eigenvalue = min(np.abs(n))
    conditionNumber = max_eigenvalue/min_eigenvalue
    return conditionNumber

def jordan_inverse(matrix):
    A = np.copy(matrix).astype(np.float64)
    n = A.shape[0]
    I = np.eye(n, dtype=np.float64)

    for i in range(n):
        factor = A[i][i]
        if(factor == 0):
            return np.linalg.inv(matrix)
        A[i] /= factor
        I[i] /= factor

        for j in range(n):
            if i != j:
                factor = A[j][i]
                A[j] = A[j] - factor * A[i]
                I[j] = I[j] - factor * I[i]

    return I

def power_method(P, iterations, tol):
    x = np.array([[1], [1], [0], [1], [0]])
    A = P.copy()
    modified_x = x
    eigen_value = 0

    for i in range(iterations):
        B = modified_x
        prev_eigen_value = eigen_value
        modified_x = A @ modified_x
        index = np.argmax(np.abs(modified_x))
        eigen_value = modified_x[index]
        modified_x = modified_x / eigen_value
        if np.max(np.abs(modified_x - B)) < tol and np.abs(prev_eigen_value - eigen_value) < tol:
            break

    return eigen_value[0]

def lu_method(A, B, L, U):
    y = np.zeros((L.shape[0], 1))
    x = np.zeros((L.shape[0], 1))

    for i in range(0, L.shape[0]):
        sum = 0
        for j in range(0, i):
            sum += L[i][j]*y[j]
        y[i] = (B[i]-sum)/L[i][i]

    for i in range(U.shape[0]-1, -1, -1):
        sum = 0
        for j in range(U.shape[0]-1, i, -1):
            sum += U[i][j]*x[j]
        x[i] = (y[i]-sum)/U[i][i]

    return x

def parse_matrix_input(matrix_string, shape):
    try:
        rows = matrix_string.strip().split('\n')
        matrix = [[float(num) for num in row.strip().split()] for row in rows]
        matrix = np.array(matrix)
        
        if matrix.shape != shape:
            raise ValueError(f"Matrix must be {shape[0]}x{shape[1]}")
        
        return matrix
    except Exception as e:
        st.error(f"Error parsing matrix: {str(e)}")
        return None

def load_custom_css():
    st.markdown("""
        <style>
        /* Dark theme colors */
        :root {
            --background: #0e1117;
            --secondary-bg: #1e1e1e;
            --text: #ffffff;
            --accent: #4c8bf5;
        }

        /* Main container styling */
        .main {
            background-color: var(--background);
            color: var(--text);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-bg);
            padding: 1rem;
        }

        .sidebar .sidebar-content {
            background-color: var(--secondary-bg);
        }

        /* Matrix display styling */
        .dataframe {
            background-color: var(--secondary-bg) !important;
            color: var(--text) !important;
        }

        /* Input fields styling */
        .stTextInput > div > div {
            background-color: var(--secondary-bg);
            color: var(--text);
        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
            background-color: var(--accent);
            color: white;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--secondary-bg);
            color: var(--text);
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="GUI-1", layout="wide")
    load_custom_css()

    # Sidebar
    st.sidebar.title("Analysis Navigation")
    st.sidebar.markdown("---")

    # Define the analysis sections
    analysis_sections = {
        "LU Decomposition": False,
        "Eigenvalues": False,
        "Determinant and Uniqueness": False,
        "Condition Number": False,
        "Characteristic Polynomial": False,
        "Jordan Inverse": False,
        "Maximum and Minimum Eigenvalues": False,
        "Solution to Ax = B": False
    }

    # Create toggles in sidebar
    st.sidebar.markdown("### Select Analysis Sections")
    for section in analysis_sections:
        analysis_sections[section] = st.sidebar.checkbox(section, True)

    # Main content
    st.title("Numerical Methods Assignment-1")

    # Matrix A input
    st.header("Matrix A")
    use_default_a = st.checkbox("Use predefined Matrix A", value=True)

    default_matrix_a = """4 2 1 3 0
2 4 1 0 3
1 1 5 2 1
3 0 2 4 1
0 3 1 1 5"""

    if use_default_a:
        matrix_input = default_matrix_a
    else:
        matrix_input = st.text_area("Matrix A", value=default_matrix_a, height=200)

    matrix_a = parse_matrix_input(matrix_input, (5, 5))
    if matrix_a is not None:
        st.write("Matrix A:")
        st.write(pd.DataFrame(matrix_a))

    # Vectors B1 and B2 input
    st.header("Vectors B1 and B2")
    use_default_b = st.checkbox("Use predefined vectors B1 and B2", value=True)

    default_b1 = """9
5
3
1
2"""

    default_b2 = """1
2
32
3
4"""

    col1, col2 = st.columns(2)
    with col1:
        if use_default_b:
            b1_input = default_b1
        else:
            b1_input = st.text_area("Vector B1", value=default_b1, height=150)
        b1 = parse_matrix_input(b1_input, (5, 1))
        if b1 is not None:
            st.write("Vector B1:")
            st.write(pd.DataFrame(b1))

    with col2:
        if use_default_b:
            b2_input = default_b2
        else:
            b2_input = st.text_area("Vector B2", value=default_b2, height=150)
        b2 = parse_matrix_input(b2_input, (5, 1))
        if b2 is not None:
            st.write("Vector B2:")
            st.write(pd.DataFrame(b2))

    # Calculate button and results
    if st.button("Calculate") and matrix_a is not None and b1 is not None and b2 is not None:
        st.header("Results")

        if analysis_sections["LU Decomposition"]:
            with st.expander("LU Decomposition", expanded=True):
                y = [0]
                L, U = lu_decomposition(matrix_a, y)
                st.write("Lower Triangular Matrix (L):")
                st.write(pd.DataFrame(L))
                st.write("Upper Triangular Matrix (U):")
                st.write(pd.DataFrame(U))

        if analysis_sections["Eigenvalues"]:
            with st.expander("Eigenvalues", expanded=True):
                eigenvalues, eigenvector = eig(matrix_a)
                det = determinant(eigenvalues)
                if eigenvalues.dtype == "complex128":
                    st.warning("Cannot find eigenvalues using LU method as eigenvalues are complex")
                elif det==0:
                    st.write("eigenvalues:", eigenvalues)
                else:
                    eigenvalues = eigen_values(matrix_a, 1000, 0.000000001) - y[0]
                    st.write("Eigenvalues:", eigenvalues)

        if analysis_sections["Determinant and Uniqueness"]:
            with st.expander("Determinant and Uniqueness", expanded=True):
                det = determinant(eigenvalues)
                st.write(f"Determinant: {det}")
                if det != 0:
                    st.success("The system has a unique solution (det ≠ 0)")
                else:
                    st.warning("The system does not have a unique solution (det = 0)")

        if analysis_sections["Condition Number"]:
            with st.expander("Condition Number", expanded=True):
                cond_num = condition_number(eigenvalues)
                st.write(f"Condition Number: {cond_num}")
                if cond_num >= np.linalg.cond(sci.hilbert(5)):
                    st.warning("The chosen matrix is ill-conditioned")
                else:
                    st.success("The matrix is well-conditioned")

        if analysis_sections["Characteristic Polynomial"]:
            with st.expander("Characteristic Polynomial", expanded=True):
                coefficients = np.poly(eigenvalues)
                st.write("Coefficients of polynomial:", coefficients)
                x = sp.symbols('x')
                poly = sp.prod([x-r for r in eigenvalues])
                expand_poly = sp.expand(poly)
                st.write("Polynomial:", expand_poly)

        if analysis_sections["Jordan Inverse"]:
            with st.expander("Jordan Inverse", expanded=True):
                if det != 0:
                    INV_A = jordan_inverse(matrix_a)
                    st.write("Jordan Inverse of Matrix:")
                    st.write(pd.DataFrame(INV_A))
                else:
                    st.warning("As det=0, inverse does not exist for matrix A")

        if analysis_sections["Maximum and Minimum Eigenvalues"]:
            with st.expander("Maximum and Minimum Eigenvalues (Power Method)", expanded=True):
                max_eigen_value = power_method(matrix_a, 500, 1e-9)
                st.write(f"Maximum eigenvalue: {max_eigen_value}")
                
                if det != 0:
                    INV_A = jordan_inverse(matrix_a)
                    O = np.zeros((5, 5))
                    if not np.array_equal(INV_A, O):
                        min_eigen_value = 1/power_method(INV_A, 500, 1e-9)
                    else:
                        min_eigen_value = 1/power_method(np.linalg.inv(matrix_a), 500, 1e-9)
                    st.write(f"Minimum eigenvalue: {min_eigen_value}")

        if analysis_sections["Solution to Ax = B"]:
            with st.expander("Solutions to Ax = B", expanded=True):
                if det != 0:
                    y = [0]
                    P, L, U = sci.lu(matrix_a)
                    x1 = lu_method(matrix_a, b1, L, U)
                    x2 = lu_method(matrix_a, b2, L, U)
                    st.write("Solution vector for Ax = B1:")
                    st.write(pd.DataFrame(x1))
                    st.write("Solution vector for Ax = B2:")
                    st.write(pd.DataFrame(x2))
        st.header("Computation Information")
        st.info("""
        - Tolerance used for convergence: 1e-9
        - Maximum iterations for iterative methods: 100
        - Matrix condition analysis based on comparison with Hilbert matrix
        - All calculations performed using 64-bit floating-point precision
        """)            

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input data and try again.")
