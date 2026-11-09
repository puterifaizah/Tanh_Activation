# app_tanh.py
# Streamlit app: Visualise the Hyperbolic Tangent (Tanh) activation function

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation Function", layout="wide")

st.title("3. Hyperbolic Tangent (Tanh)")
st.write(
    """
    The **Tanh** activation function is defined as:

    $$\\tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

    It squashes any real input into the range **(−1, 1)**, and unlike Sigmoid it is
    **zero-centred**, which often makes training converge faster. Its derivative is
    $\\tanh'(x) = 1 - \\tanh^2(x)$.
    """
)

# --- Sidebar controls ---
st.sidebar.header("Settings")
x_min = st.sidebar.slider("x-axis minimum", -20.0, 0.0, -6.0, 0.5)
x_max = st.sidebar.slider("x-axis maximum", 0.0, 20.0, 6.0, 0.5)
n_points = st.sidebar.slider("Number of points", 100, 2000, 400, 50)
show_derivative = st.sidebar.checkbox("Show derivative", value=False)

# --- Compute values ---
x = np.linspace(x_min, x_max, n_points)
y = np.tanh(x)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="seagreen", linewidth=2.5,
        label="tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)")

if show_derivative:
    dy = 1 - y ** 2
    ax.plot(x, dy, color="purple", linewidth=2, linestyle="--",
            label="tanh'(x) = 1 − tanh²(x)")

ax.axhline(0, color="black", linewidth=0.6)
ax.axhline(1, color="grey", linewidth=0.6, linestyle=":")
ax.axhline(-1, color="grey", linewidth=0.6, linestyle=":")
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Tanh Activation Function")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

st.pyplot(fig)

# --- Property table ---
st.subheader("Properties")
st.table({
    "Property": ["Range", "Derivative", "Zero-centred?", "Vanishing gradient?"],
    "Value": [
        "(−1, 1)",
        "1 − tanh²(x)",
        "Yes",
        "Yes (for large |x|)",
    ],
})

# --- Interactive single-value calculator ---
st.subheader("Try a value")
xv = st.number_input("Enter x", value=0.0, step=0.1)
st.write(f"**Tanh({xv}) = {np.tanh(xv):.6f}**")