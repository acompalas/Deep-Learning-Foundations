import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from PIL import Image

# -------------------------------
# Utility Functions
# -------------------------------
def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if isinstance(v, (float, int)) else "None"

def generate_synthetic_data(n_points=1000):
    x = np.random.uniform(0, 10, n_points)
    true_slope = np.random.uniform(-2, 2)
    true_intercept = np.random.uniform(0, 10)
    noise = np.random.normal(0.0, 2.0, size=x.shape)
    y = true_slope * x + true_intercept + noise
    rss_true = np.sum(noise**2)
    return {
        "x": x,
        "y": y,
        "noise": noise,
        "true_slope": true_slope,
        "true_intercept": true_intercept,
        "noise_std": 2.0,
        "rss_true": rss_true
    }

def crop_and_resize(img, target_size=(500, 500)):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    if img_ratio > target_ratio:
        new_width = int(target_ratio * img.height)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))
    return img.resize(target_size, Image.Resampling.LANCZOS)

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Ordinary Least Squares")

st.header("Overview")
st.markdown("""
Ordinary Least Squares (OLS) is a closed-form analytical approach to Linear Regression which models the relationship between a set of features and a continuous target variable.
""")

st.header("Linear Regression")
st.markdown("We assume a linear relationship of the form:")
st.latex(r"y = \beta_0 + \beta_1 x + \varepsilon")

st.markdown(r"""
Where:
- $y$ is the **response variable**  
- $x$ is the **predictor variable**  
- $\beta_0$ is the **intercept**  
- $\beta_1$ is the **slope**  
- $\varepsilon$ is the **error term**
""")


st.subheader("Slope Intercept Model vs. Linear Regression")

st.markdown("""
Linear regression is closely related to the **slope-intercept model** from algebra, which is purely deterministic.

However, in statistics, **linear regression is stochastic**, meaning that we assume the data contains some inherent randomness:
""")

# -------------------------------
# Images: slope-intercept vs. noisy fit
# -------------------------------
img1 = Image.open("assets/images/slope_intercept.png")
img2 = Image.open("assets/images/noisy_fit.png")
col1, col2 = st.columns(2)
with col1:
    st.image(crop_and_resize(img1), caption="Deterministic Slope-Intercept Model")
with col2:
    st.image(crop_and_resize(img2), caption="Stochastic Linear Regression Fit")

st.latex(r"\mathbb{E}[\varepsilon] = 0 \quad\quad \varepsilon \sim \mathcal{N}(0, \sigma)")

# -------------------------------
# Model Type Selection
# -------------------------------
st.subheader("Types of Linear Regression Models")

model_type = st.radio(
    "Select a model type to view its formulation:",
    ["Simple Linear Regression", "Multiple Linear Regression", "Multivariate Linear Regression"]
)

if model_type == "Simple Linear Regression":
    st.latex(r"\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x")
    st.markdown("This is the general form of **simple linear regression** with a single predictor variable.")

elif model_type == "Multiple Linear Regression":
    st.latex(r"\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + \dots + \hat{\beta}_n x_n = \boldsymbol{\beta}^\top \mathbf{x}")
    st.latex(r"""
    \boldsymbol{\beta} =
    \begin{bmatrix}
    \hat{\beta}_0 \\
    \hat{\beta}_1 \\
    \hat{\beta}_2 \\
    \vdots \\
    \hat{\beta}_n
    \end{bmatrix}
    \quad
    \mathbf{x} =
    \begin{bmatrix}
    1 \\
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
    \end{bmatrix}
    """)

elif model_type == "Multivariate Linear Regression":
    st.latex(r"\mathbf{\hat{y}} = \mathbf{X} \boldsymbol{\beta}")
    st.markdown("For example, with 3 data points and 2 features (plus a bias term):")
    st.latex(r"""
    \mathbf{\hat{y}} =
    \begin{bmatrix}
    \hat{y}_0 \\
    \hat{y}_1 \\
    \hat{y}_2
    \end{bmatrix}
    \quad
    \mathbf{X} =
    \begin{bmatrix}
    1 & x_{11} & x_{12} \\
    1 & x_{21} & x_{22} \\
    1 & x_{31} & x_{32}
    \end{bmatrix}
    \quad
    \boldsymbol{\beta} =
    \begin{bmatrix}
    \hat{\beta}_0 \\
    \hat{\beta}_1 \\
    \hat{\beta}_2
    \end{bmatrix}
    """)

# -------------------------------
# Generate Synthetic Data
# -------------------------------
st.subheader("Generate Synthetic Data")
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False

chart_placeholder = st.empty()
generate_clicked = st.button("Generate Data")

if generate_clicked:
    st.session_state.synthetic_data = generate_synthetic_data()
    st.session_state.data_generated = True

# -------------------------------
# Plot Data
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
if st.session_state.data_generated:
    data = st.session_state.synthetic_data
    x, y = data["x"], data["y"]
    ax.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6, label='Noisy Observations')
    ax.set_title("Randomly Generated Synthetic Data")
    ax.legend()
else:
    ax.set_title("Click 'Generate Data' to display a plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
chart_placeholder.pyplot(fig)


# -------------------------------
# Show Parameters
# -------------------------------
data = st.session_state.get("synthetic_data", {})
st.markdown("#### True Model Parameters")
st.markdown(f"- True slope (β₁): **{fmt(data.get('true_slope'))}**")
st.markdown(f"- True intercept (β₀): **{fmt(data.get('true_intercept'))}**")
st.markdown(f"- Noise standard deviation (σ): **2.0000**")
st.markdown(f"- Total Noise (∑ε²): **{fmt(data.get('rss_true'))}**")

# -------------------------------
# RSS Interactive Fit
# -------------------------------
st.subheader("Residual Sum-of Squares (RSS)")
st.markdown("The goal of Linear Regression is to minimize the total sum of squared errors")
st.latex(r"RSS = \sum_{i=1}^m (y_i - \hat{y}_i)^2 = \sum_{i=1}^m (y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i))^2")
st.latex(r"RSS = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})")
st.markdown("This represents the distance between the model predictions and the actual data.")

# -------------------------------
# Interactive Visualizations
# -------------------------------
st.subheader("Try optimizing manually.")
if st.session_state.data_generated:
    mode = st.radio("Visualization Mode", ["Manual Fit", "Optimize One Parameter", "RSS Surface + Contour"])

    x = data["x"]
    y = data["y"]
    X_full = np.c_[np.ones_like(x), x]
    y_full = y.reshape(-1, 1)

    # Initialize sliders with resettable defaults
    default_b0 = float(np.mean(y))
    default_b1 = 0.0

    if "b0" not in st.session_state:
        st.session_state.b0 = default_b0
    if "b1" not in st.session_state:
        st.session_state.b1 = default_b1

    # Sliders
    st.session_state.b0 = st.slider("β₀", -20.0, 20.0, st.session_state.b0, 0.05)
    st.session_state.b1 = st.slider("β₁", -5.0, 5.0, st.session_state.b1, 0.05)

    # RSS Calculation
    b0 = st.session_state.b0
    b1 = st.session_state.b1
    beta = np.array([[b0], [b1]])
    y_hat = X_full @ beta
    rss = np.sum((y_full - y_hat) ** 2)

    # Feedback Message
    rss_diff = abs(rss - data["rss_true"])
    if rss_diff < 100:
        st.success(f"Great job! Your RSS is very close to the true noise level (Δ = {rss_diff:.2f})")
    else:
        st.info(f"Keep adjusting! Your RSS is still off by about {rss_diff:.2f} from the true noise")

    # Reset Button
    if st.button("Reset Sliders to Default"):
        st.session_state.b0 = default_b0
        st.session_state.b1 = default_b1
        st.rerun()()

    beta_range = np.linspace(-20, 20, 300)

    def plot_manual():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label=f"RSS: {rss:.2f}")
        ax.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax.set_title("Manual Fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        return fig

    def plot_optimize_one():
        rss_beta0 = [np.sum((y_full - (X_full @ np.array([[b], [b1]]))) ** 2) for b in beta_range]
        rss_beta1 = [np.sum((y_full - (X_full @ np.array([[b0], [b]]))) ** 2) for b in beta_range]
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1])
        # Left plot: fit line
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax0.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label=f"RSS: {rss:.2f}")
        ax0.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax0.set_title("Data Fit")
        ax0.legend()
        ax0.grid(True)

        # Top-right: RSS vs β₀
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(beta_range, rss_beta0)
        ax1.plot(b0, rss, 'ro', label=f"β₀ = {b0:.2f}")
        ax1.set_xlim(-20, 20)  # Match slider for β₀
        ax1.set_title("RSS vs β₀")
        ax1.legend()
        ax1.grid(True)

        # Bottom-right: RSS vs β₁
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(beta_range, rss_beta1)
        ax2.plot(b1, rss, 'ro', label=f"β₁ = {b1:.2f}")
        ax2.set_xlim(-5, 5)  # Match slider for β₁
        ax2.set_title("RSS vs β₁")
        ax2.legend()
        ax2.grid(True)

        return fig

    def plot_surface_contour():
        b0_range = np.linspace(-5, 15, 100)
        b1_range = np.linspace(-3, 5, 100)
        B0, B1 = np.meshgrid(b0_range, b1_range)
        RSS_surface = np.array([
            np.sum((y_full - X_full @ np.array([[b0], [b1]]))**2)
            for b0, b1 in zip(np.ravel(B0), np.ravel(B1))
        ]).reshape(B0.shape)

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

        # Data fit plot
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax0.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label=f"RSS: {rss:.2f}")
        ax0.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax0.set_title("Fit")
        ax0.legend()
        ax0.grid(True)

        # 3D surface plot
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax1.plot_surface(B0, B1, RSS_surface, cmap=cm.viridis, alpha=0.9)
        ax1.scatter(b0, b1, rss, color='red', s=50)
        ax1.set_title("RSS Surface")

        # Contour line plot (not filled)
        ax2 = fig.add_subplot(gs[1, 1])
        cp = ax2.contour(B0, B1, RSS_surface, levels=30, cmap=cm.viridis)
        ax2.plot(b0, b1, 'ro')
        ax2.set_title("RSS Contour Lines")
        ax2.set_xlabel(r"$\beta_0$")
        ax2.set_ylabel(r"$\beta_1$")
        fig.colorbar(cp, ax=ax2)

        return fig


    # Render the selected visualization
    if mode == "Manual Fit":
        st.pyplot(plot_manual())
    elif mode == "Optimize One Parameter":
        st.pyplot(plot_optimize_one())
    elif mode == "RSS Surface + Contour":
        st.pyplot(plot_surface_contour())
else:
    st.warning("Please generate data before fitting.")
        
st.header("Ordinary Least Squares")
st.markdown(r"""
We assume the linear model:
$$\hat{y} = \beta_0 + \beta_1 x = X \boldsymbol{\beta}$$

Where:
$$
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}
\in \mathbb{R}^{n \times 2},
\quad
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0 \\
\beta_1
\end{bmatrix}
\in \mathbb{R}^{2 \times 1}
$$

We want to minimize the residual sum of squares (RSS):
$$
\text{RSS} = \| \mathbf{y} - X\boldsymbol{\beta} \|^2 = (\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})
$$

Setting the gradient to zero:
$$
\frac{\partial}{\partial \boldsymbol{\beta}} \left( \mathbf{y} - X\boldsymbol{\beta} \right)^T \left( \mathbf{y} - X\boldsymbol{\beta} \right) = 0
$$

Solving:
$$
X^T X \boldsymbol{\beta} = X^T \mathbf{y} \\
\Rightarrow \boldsymbol{\beta} = (X^T X)^{-1} X^T \mathbf{y}
$$
""")

# -------------------------------
# Fit with OLS
# -------------------------------
st.header("Fit with OLS")

if st.session_state.get("data_generated", False):
    x = data["x"]
    y = data["y"]
    X = np.c_[np.ones_like(x), x]
    y_vec = y.reshape(-1, 1)

    # Only initialize b0 and b1 if not already set
    if "b0" not in st.session_state or "b1" not in st.session_state:
        st.session_state.b0 = float(np.mean(y))
        st.session_state.b1 = 0.0

    # If ols_fit has run, override betas with solution
    if st.session_state.get("ols_fit", False):
        st.session_state.b0 = st.session_state.get("b0_hat", st.session_state.b0)
        st.session_state.b1 = st.session_state.get("b1_hat", st.session_state.b1)

    b0 = st.session_state.b0
    b1 = st.session_state.b1
    y_hat = X @ np.array([[b0], [b1]])
    rss = float(np.sum((y_vec - y_hat) ** 2))

    ols_mode = st.radio(
        "Select OLS Visualization Mode",
        ["OLS Fit Line", "OLS + RSS vs Each β", "OLS on Surface & Contour"]
    )

    st.markdown(f"""
    **Intercepts:**   β₀ = {data['true_intercept']:.4f}, β̂₀ = {b0:.4f}  
    **Slopes:**    β₁ = {data['true_slope']:.4f}, β̂₁ = {b1:.4f}  
    **True Noise (∑ε²):** {fmt(data.get('rss_true'))}  
    **Fitted RSS:**    {rss:.2f}  
    **|RSS − ∑ε²|:**    {abs(rss - data['rss_true']):.2f}
    """)
    beta_range = np.linspace(-20, 20, 300)

    def plot_ols_fit():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label="Current Fit")
        ax.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax.set_title("Fit Line vs True Line")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        return fig

    def plot_ols_optimize_one():
        rss_beta0 = [np.sum((y_vec - (X @ np.array([[b], [b1]]))) ** 2) for b in beta_range]
        rss_beta1 = [np.sum((y_vec - (X @ np.array([[b0], [b]]))) ** 2) for b in beta_range]
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1])

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax0.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label=f"Fit (RSS: {rss:.2f})")
        ax0.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax0.set_title("Current Fit")
        ax0.legend()
        ax0.grid(True)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(beta_range, rss_beta0)
        ax1.plot(b0, rss, 'ro', label=f"β₀ = {b0:.2f}")
        ax1.set_xlim(-20, 20)
        ax1.set_title("RSS vs β₀")
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(beta_range, rss_beta1)
        ax2.plot(b1, rss, 'ro', label=f"β₁ = {b1:.2f}")
        ax2.set_xlim(-5, 5)
        ax2.set_title("RSS vs β₁")
        ax2.legend()
        ax2.grid(True)

        return fig

    def plot_ols_surface_contour():
        b0_range = np.linspace(-5, 15, 100)
        b1_range = np.linspace(-3, 5, 100)
        B0, B1 = np.meshgrid(b0_range, b1_range)
        RSS_surface = np.array([
            np.sum((y_vec - X @ np.array([[b0], [b1]])) ** 2)
            for b0, b1 in zip(np.ravel(B0), np.ravel(B1))
        ]).reshape(B0.shape)

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6)
        ax0.plot(np.sort(x), y_hat[np.argsort(x)], 'k-', label=f"Fit (RSS: {rss:.2f})")
        ax0.plot(np.sort(x), data["true_slope"] * np.sort(x) + data["true_intercept"], 'k--', label="True Line")
        ax0.set_title("Current Fit")
        ax0.legend()
        ax0.grid(True)

        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax1.plot_surface(B0, B1, RSS_surface, cmap=cm.viridis, alpha=0.9)
        ax1.scatter(b0, b1, rss, color='red', s=50)
        ax1.set_title("RSS Surface")

        ax2 = fig.add_subplot(gs[1, 1])
        cp = ax2.contour(B0, B1, RSS_surface, levels=30, cmap=cm.viridis)
        ax2.plot(b0, b1, 'ro')
        ax2.set_title("RSS Contour")
        ax2.set_xlabel(r"$\beta_0$")
        ax2.set_ylabel(r"$\beta_1$")
        fig.colorbar(cp, ax=ax2)

        return fig

    # Render the selected plot
    if ols_mode == "OLS Fit Line":
        st.pyplot(plot_ols_fit())
    elif ols_mode == "OLS + RSS vs Each β":
        st.pyplot(plot_ols_optimize_one())
    elif ols_mode == "OLS on Surface & Contour":
        st.pyplot(plot_ols_surface_contour())

    # Buttons after chart
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Run OLS Fit"):
            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_vec
            st.session_state.b0_hat = float(beta_hat[0, 0])
            st.session_state.b1_hat = float(beta_hat[1, 0])
            st.session_state.ols_fit = True
            st.rerun()

    with col2:
        if st.button("Reset Betas"):
            st.session_state.b0 = float(np.mean(y))
            st.session_state.b1 = 0.0
            st.session_state.ols_fit = False
            st.rerun()

else:
    st.warning("Please generate data before fitting.")


# -------------------------------
# R² Test Section
# -------------------------------
st.markdown(r"""
### $R^2$-test

The $R^2$ score measures the proportion of variance in the dependent variable that is predictable from the independent variable(s)

Let $y$ be the true values and $\hat{y}$ be the predicted values. Then:

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

$$
RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2,\quad
TSS = \sum_{i=1}^n (y_i - \bar{y})^2
$$

- $RSS$: Residual Sum of Squares  
- $TSS$: Total Sum of Squares  
- $R^2 = 1$: perfect fit  
- $R^2 = 0$: model no better than mean  
- $R^2 < 0$: model worse than mean predictor
""")

# Show button and conditionally compute score
if st.button("Run R² Test"):
    if "data" in locals() and st.session_state.get("ols_fit", False):
        # Use stored OLS-fit parameters
        b0 = st.session_state.get("b0_hat", st.session_state.b0)
        b1 = st.session_state.get("b1_hat", st.session_state.b1)

        # Ensure shapes are consistent
        x = data["x"]
        y_true = data["y"]
        X = np.c_[np.ones_like(x), x]
        beta_hat = np.array([[b0], [b1]])
        y_pred = (X @ beta_hat).flatten()

        rss_val = np.sum((y_true - y_pred) ** 2)
        tss_val = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - rss_val / tss_val

        st.markdown(f"""
        ### $R^2$ Test Results  
        **RSS:** {rss_val:.2f}  
        **TSS:** {tss_val:.2f}  
        **R² Score:**
        {r2_score:.4f}
        """)
        st.success(f"Your model explains approximately {r2_score * 100:.2f}% of the variance in the response variable.")
    else:
        st.warning("Please run OLS Fit before testing R².")


