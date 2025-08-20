import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from utils.regularization import RidgeRegression

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
    


st.title("Introduction to Regularization")

section = st.selectbox("", [
    "Overview",
    "Bayesian Perspective",
    "Method Comparisons",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
])

if section == "Overview":
    st.markdown(r"""
    ## Regularization Overview

    ### The Bias-Variance Tradeoff

    In machine learning, there's a fundamental tradeoff between:

    - **Bias** – Error from overly simplistic models that fail to capture the true relationship in the data  
    - **Variance** – Error from overly complex models that are sensitive to small fluctuations in the training set and fail to generalize.

    A **linear model on nonlinear data** has high bias but low variance. A **very flexible model** (like a high-degree polynomial) may fit training data perfectly but perform poorly on new data due to high variance.

    ---
    """)
    
    # First image: Bullseye Intuition
    st.image("assets/images/bullseye.png", caption="🎯 Bias-Variance Intuition (Bullseye Diagram)", use_container_width=True)
    st.markdown("""
    This diagram visualizes **bias and variance** using a target metaphor:

    - **Top-left**: Low bias, low variance — predictions are tightly clustered around the bullseye (ideal).
    - **Top-right**: Low bias, high variance — predictions center on the target but are spread out.
    - **Bottom-left**: High bias, low variance — consistent, but far from the true target.
    - **Bottom-right**: High bias, high variance — inconsistent and inaccurate.

    Regularization aims to shift us closer to the **top-left quadrant**.
    """)

    # Second image: Bias-Variance Decomposition Curve
    st.image("assets/images/biasvariance.png", caption="📈 Error Decomposition vs Model Complexity", use_container_width=True)
    st.markdown("""
    As model complexity increases:

    - **Bias decreases**: The model becomes more flexible.
    - **Variance increases**: The model overfits to noise in the training data.

    **Total error** is minimized at a sweet spot — this is where **regularization** helps by pulling us back from overfitting.

    """)    
    
    st.markdown(r"""

    ### Dealing with the Tradeoff

    There are several techniques to manage the bias-variance tradeoff:

    - **Regularization** – Penalizes model complexity by shrinking coefficients (our focus here)  
    - **Bagging** – Reduces variance by averaging over multiple models (e.g., random forests)  
    - **Boosting** – Sequentially reduces bias by combining weak learners

    ---

    ### 📌 In This Module

    We’ll focus on **regularization**, specifically:

    - **Ridge Regression (L2)**  
    Adds a penalty proportional to the **sum of squared weights**:  
    $$
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \text{Cost}(y_i, \hat{y}_i) + \lambda \sum_{j=1}^p w_j^2
    $$

    - **Lasso Regression (L1)**  
    Adds a penalty proportional to the **sum of absolute weights**:  
    $$
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \text{Cost}(y_i, \hat{y}_i) + \lambda \sum_{j=1}^p |w_j|
    $$

    - **Elastic Net (L1 + L2)**  
    Combines both penalties for a flexible tradeoff between **shrinkage** and **sparsity**:  
    $$
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \text{Cost}(y_i, \hat{y}_i) + \lambda_1 \sum_{j=1}^p |w_j| + \lambda_2 \sum_{j=1}^p w_j^2
    $$

    ---

    You’ll learn:

    - When and why to use each
    - How they impact model behavior
    - Visual and code examples
    """)
    
if section == "Bayesian Perspective":
    st.markdown("## Bayesian Perspective on Regularization")

    st.markdown("""
    ### Why Think of Weights as Distributions?

    In the Bayesian framework, we don’t treat weights as fixed numbers — we treat them as **random variables** drawn from a **prior distribution**.  
    This allows us to encode assumptions like:

    - **Small weights are more likely** (i.e., we expect the model to be simple)
    - **Some weights might be exactly zero** (Lasso)

    This probabilistic viewpoint allows us to regularize models by incorporating **prior knowledge** into the learning process.
    """)

    st.markdown("""
    ### Bayes’ Theorem Refresher

    Bayes' rule gives us a way to compute the **posterior** over weights given the data:

    $$
    P(\\theta \\mid \\mathcal{D}) = \\frac{P(\\mathcal{D} \\mid \\theta) \\cdot P(\\theta)}{P(\\mathcal{D})}
    $$

    - $P(\\mathcal{D} \\mid \\theta)$ is the **likelihood** — how well the weights explain the data.
    - $P(\\theta)$ is the **prior** — what we believe about the weights before seeing data.
    - $P(\\theta \\mid \\mathcal{D})$ is the **posterior** — our updated belief after seeing data.
    - $P(\\mathcal{D})$ is the **evidence**, which we ignore in optimization since it doesn’t depend on $\\theta$.
    """)

    st.markdown("""
    ### MAP Estimation

    In Maximum A Posteriori (MAP) estimation, we find the weights $\\theta$ that **maximize the posterior**:

    $$
    \\theta^* = \\arg\\max_\\theta \\; P(\\mathcal{D} \\mid \\theta) \\cdot P(\\theta)
    $$

    Taking the **negative log** to turn this into a loss minimization problem:

    $$
    \\mathcal{L}(\\theta) = \\arg\\min_\\theta (-\\log P(\\mathcal{D} \\mid \\theta) - \\log P(\\theta))
    $$

    This is exactly how **regularized loss functions** are formed.
    """)

    st.markdown("---")

    st.markdown("### Deriving Ridge Regression (L2) from a Gaussian Prior")

    st.markdown(r"""
    We assume:

    **Data likelihood** (assuming i.i.d. Gaussian noise):
    $$
    P(\mathcal{D} \mid \theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_y^2}} 
    \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right)
    $$

    **Weight prior** (zero-mean Gaussian for each $\theta_j$):
    $$
    P(\theta) = \prod_{j=1}^p \frac{1}{\sqrt{2\pi\sigma_\theta^2}} 
    \exp\left( -\frac{\theta_j^2}{2\sigma_\theta^2} \right)
    $$

    ---

    We perform **MAP estimation**, i.e. maximize the posterior:

    $$
    \theta^* = \text{argmax}_\theta \; P(\mathcal{D} \mid \theta) \cdot P(\theta)
    $$

    Taking the **negative log** turns this into a loss minimization:

    $$
    \mathcal{L}(\theta) = -\log P(\mathcal{D} \mid \theta) - \log P(\theta)
    $$

    ---

    ### Step-by-Step:

    $$
    \begin{aligned}
    \theta^* &= \arg\max \ \log \left( P(\text{Data} \mid \theta) \cdot P(\theta) \right) \\
    \\
    &= \arg\max \ \log\left( 
    \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_y^2}} 
    \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right) 
    \cdot 
    \prod_{j=1}^p \frac{1}{\sqrt{2\pi\sigma_\theta^2}} 
    \exp\left( -\frac{\theta_j^2}{2\sigma_\theta^2} \right)
    \right) \\
    \\
    &= \arg\max \ \sum_{i=1}^{N} \log \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right) 
    + \sum_{j=1}^p \log \exp\left( -\frac{\theta_j^2}{2\sigma_\theta^2} \right) \\
    \\
    &= \arg\max \ \left( -\frac{1}{\sigma_y^2} \sum_{i=1}^N (y_i - x_i^\top \theta)^2 
    - \frac{1}{\sigma_{\theta}^2} \sum_{j=1}^p \theta_j^2 \right) \\
    \\
    &= \arg\min \ \sum_{i=1}^N (y_i - x_i^\top \theta)^2 + \lambda \sum_{j=1}^p \theta_j^2 \\
    \\
    &\text{where } \lambda = \frac{\sigma_y^2}{\sigma_\theta^2}
    \end{aligned}
    $$

    ---

    ### Final Form

    This is the classic **Ridge Regression** loss:

    $$
    \mathcal{L}(\theta) = \text{MSE Loss} + \lambda \|\theta\|_2^2
    $$

    - Larger $\lambda$ = stronger prior (small $\sigma_\theta^2$) → smaller weights  
    - Smaller $\lambda$ = more trust in data (small $\sigma_y^2$) → more flexible model
    """)

    st.markdown("---")
    
    # Sliders
    sigma_y = st.slider("Data noise (σ_y)", 0.1, 5.0, 1.0, 0.1, key="ridge_sigma_y")
    sigma_theta = st.slider("Weight prior spread (σ_θ)", 0.1, 5.0, 1.0, 0.1, key="ridge_sigma_theta")


    # λ value
    lambda_val = round((sigma_y ** 2) / (sigma_theta ** 2), 2)

    # X and distributions
    x_vals = np.linspace(-6, 6, 500)
    data_dist = (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(-x_vals**2 / (2 * sigma_y**2))
    theta_dist = (1 / (np.sqrt(2 * np.pi) * sigma_theta)) * np.exp(-x_vals**2 / (2 * sigma_theta**2))

    st.latex(r"\lambda = \frac{\sigma_y^2}{\sigma_\theta^2} = " + str(lambda_val))

    # Create columns
    col1, col2 = st.columns(2)

    # Plot 1: Data Likelihood
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=x_vals, y=data_dist,
            fill='tozeroy',
            name="Data Likelihood",
            line_color='red'
        ))
        fig1.update_layout(
            title=f"Data Likelihood (σ_y = {sigma_y})",
            xaxis_title="y",
            yaxis_title="Density",
            template="plotly",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True, key="ridge_fig_data")

    # Plot 2: Weight Prior
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_vals, y=theta_dist,
            fill='tozeroy',
            name="Weight Prior",
            line_color='purple'
        ))
        fig2.update_layout(
            title=f"Weight Prior (σ_θ = {sigma_theta})",
            xaxis_title="θ",
            yaxis_title="Density",
            template="plotly",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True, key="ridge_fig_prior")

    st.markdown("### Deriving Lasso Regression (L1) from a Laplacian Prior")

    st.markdown(r"""
    We assume:

    **Data likelihood** (i.i.d. Gaussian noise, same as Ridge):
    $$
    P(\mathcal{D} \mid \theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_y^2}} 
    \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right)
    $$

    **Weight prior** (zero-mean Laplace distribution for each $\theta_j$):
    $$
    P(\theta) = \prod_{j=1}^p \frac{1}{2b} 
    \exp\left( -\frac{|\theta_j|}{b} \right)
    $$

    ---

    We perform **MAP estimation**, i.e. maximize the posterior:

    $$
    \theta^* = \text{argmax}_\theta \; P(\mathcal{D} \mid \theta) \cdot P(\theta)
    $$

    Taking the **negative log** turns this into a loss minimization:

    $$
    \mathcal{L}(\theta) = -\log P(\mathcal{D} \mid \theta) - \log P(\theta)
    $$

    ---

    ### Step-by-Step:

    $$
    \begin{aligned}
    \theta^* &= \arg\max \ \log \left( P(\text{Data} \mid \theta) \cdot P(\theta) \right) \\
    \\
    &= \arg\max \ \log\left( 
    \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_y^2}} 
    \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right) 
    \cdot 
    \prod_{j=1}^p \frac{1}{2b} 
    \exp\left( -\frac{|\theta_j|}{b} \right)
    \right) \\
    \\
    &= \arg\max \ \sum_{i=1}^{N} \log \exp\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma_y^2} \right) 
    + \sum_{j=1}^p \log \exp\left( -\frac{|\theta_j|}{b} \right) \\
    \\
    &= \arg\max \ \left( -\frac{1}{\sigma_y^2} \sum_{i=1}^N (y_i - x_i^\top \theta)^2 
    - \frac{1}{b} \sum_{j=1}^p |\theta_j| \right) \\
    \\
    &= \arg\min \ \sum_{i=1}^N (y_i - x_i^\top \theta)^2 + \lambda \sum_{j=1}^p |\theta_j| \\
    \\
    &\text{where } \lambda = \frac{\sigma_y^2}{b}
    \end{aligned}
    $$

    ---

    ### Final Form

    This is the classic **Lasso Regression** loss:

    $$
    \mathcal{L}(\theta) = \text{MSE Loss} + \lambda \|\theta\|_1
    $$

    - Larger $\lambda$ = stronger belief in **sparsity** (small $b$ in Laplace prior)  
    - Smaller $\lambda$ = more trust in data (small $\sigma_y^2$) → more flexibility in weights  
    - Lasso **encourages exact zeros** in $\theta_j$ due to the sharp corners of the L1 penalty
    """)

    # Sliders
    sigma_y = st.slider("Data noise (σ_y)", 0.1, 5.0, 1.0, 0.1, key="lasso_sigma_y")
    b = st.slider("Weight prior spread (Laplace scale b)", 0.1, 5.0, 1.0, 0.1, key="lasso_b")

    # Compute lambda
    lambda_val = round((sigma_y ** 2) / b, 2)

    # X-axis range
    x_vals = np.linspace(-6, 6, 500)

    # Distributions
    data_dist = (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(-x_vals**2 / (2 * sigma_y**2))
    theta_laplace = (1 / (2 * b)) * np.exp(-np.abs(x_vals) / b)
    
    st.latex(r"\lambda = \frac{\sigma_y^2}{b} = " + str(lambda_val))

    # Columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Plot 1: Data Likelihood
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=x_vals, y=data_dist,
            fill='tozeroy',
            name="Data Likelihood",
            line_color='red'
        ))
        fig1.update_layout(
            title=f"Data Likelihood (σ_y = {sigma_y})",
            xaxis_title="y",
            yaxis_title="Density",
            template="plotly",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True, key="lasso_fig_data")

    # Plot 2: Weight Prior (Laplace in green)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_vals, y=theta_laplace,
            fill='tozeroy',
            name="Weight Prior (Laplace)",
            line_color='green'
        ))
        fig2.update_layout(
            title=f"Weight Prior (b = {b})",
            xaxis_title="θ",
            yaxis_title="Density",
            template="plotly",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True, key="lasso_fig_prior")

if section == "Method Comparisons":
    if section == "Method Comparisons":
        st.header("Method Comparisons")

        st.markdown(r"""
        Regularization techniques modify the **optimization objective** by introducing a penalty on model complexity.
        
        Instead of simply minimizing the raw objective function, regularized regression minimizes a **penalized loss**:
        
        $$
        \mathcal{L}(\theta) = \text{MSE Loss} + \lambda \cdot \text{Penalty}(\theta)
        $$
        
        The penalty term determines how much we **discourage large or complex weights**.
        """)

        st.image("assets/images/reg_methods.png", caption="Level curves for L2 (Ridge), L1 (Lasso), and Elastic Net")

        st.markdown("### 📎 Constrained Interpretation")

        st.markdown(r"""
        Another way to view regularization is as a **constrained optimization** problem:

        $$
        \min_\theta \ \text{MSE Loss} \quad \text{subject to} \quad \|\theta\| \leq t
        $$

        - The loss is minimized **within a constraint region** defined by the penalty.
        - The shape of the region (circle, diamond, etc.) determines **which directions** are favored or suppressed.

        The **optimal weights without regularization** usually lie outside this region. Regularization **forces the solution to land on the boundary** of the constraint set instead of at the optimal solution.
        """)

        st.markdown("### Ridge Regression (L2)")

        st.latex(r"\mathcal{L}(\theta) = \text{MSE Loss} + \lambda \|\theta\|_2^2")

        st.markdown("""
        - Constraint region: **circle** (L2 ball)  
        - Encourages **small weights**, but typically **does not produce exact zeros**
        - Distributes weight **smoothly across features**
        """)

        st.markdown("### Lasso Regression (L1)")

        st.latex(r"\mathcal{L}(\theta) = \text{MSE Loss} + \lambda \|\theta\|_1")

        st.markdown("""
        - Constraint region: **diamond** (L1 ball)  
        - Encourages **sparsity**: many weights become **exactly zero**  
        - Especially useful for **feature selection**
        """)

        st.markdown("### Elastic Net")

        st.latex(r"\mathcal{L}(\theta) = \text{MSE Loss} + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2")

        st.markdown("""
        - Combines the strengths of both Lasso and Ridge  
        - Useful when features are **correlated**: Lasso alone may randomly drop one  
        - Encourages **grouped sparsity + stability**
        """)

        st.markdown("### 🔍 Summary")

        st.markdown("""
        | Method        | Penalty Shape | Encourages     | Zero Weights? | Best For |
        |---------------|----------------|----------------|---------------|----------|
        | Ridge (L2)    | Circle         | Small weights  | ❌ No          | Stable models with many small effects |
        | Lasso (L1)    | Diamond        | Sparsity       | ✅ Yes         | Feature selection |
        | Elastic Net   | Combo          | Sparse + stable| ✅ Often       | Correlated features |
        """)

        st.info("👉 Use Lasso when you expect many irrelevant features. Use Ridge when you expect all features to contribute a little. Use Elastic Net when features are correlated.")

if section == "Ridge Regression":
    st.header("Ridge Regression (L2)")
    
    st.markdown("""
                
    Ridge regression is a type of linear regression that includes L2 regularization, adding a penalty term based on the square of the coefficients to the linear regression cost function.

    Recall the Ridge Regression loss derived from **Maximum A Posteriori (MAP) estimation**:

    $$
    \\mathcal{L}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N (y_i - x_i^T \\theta)^2 + \\lambda \\|\\theta\\|_2^2
    $$

    - The first term encourages the model to fit the data.
    - The second term **penalizes large weights**, pulling $\\theta$ toward zero.
    - The strength of the penalty is controlled by $\\lambda$.

    Now let’s visualize how this regularization shapes the optimization.
    """)
    
    # --- Section Header ---
    st.subheader("📈 Generate Synthetic Data for Ridge Regression")
    
    st.markdown("""
    We generate synthetic data for Ridge Regression sampling from a normal distribution with random variance.            
    """)

    # --- Generate Button and Session Management ---
    if "data_generated" not in st.session_state:
        st.session_state.data_generated = False

    chart_placeholder = st.empty()
    generate_clicked = st.button("🔄 Generate New Data")

    if generate_clicked:
        st.session_state.synthetic_data = generate_synthetic_data()
        st.session_state.data_generated = True

    # --- Plotting the synthetic data ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if st.session_state.data_generated:
        data = st.session_state.synthetic_data
        x, y = data["x"], data["y"]
        ax.plot(x, y, 'o', ms=5, mec='k', color='red', alpha=0.6, label='Noisy Observations')
        ax.set_title("Randomly Generated Synthetic Data")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
    else:
        ax.set_title("Click 'Generate New Data' to display a plot")

    chart_placeholder.pyplot(fig)
    
    st.subheader("✂️ Train-Test Split")
    
    st.markdown("""
    We shuffle the synthetic data and split the datapoints into train and test sets.

    To better illustrate the **impact of regularization**, we should intentionally choose a **smaller training set** and a **larger test set**.

    - With fewer training examples, the model is more likely to **overfit** — especially without regularization.
    - A larger test set helps us **evaluate generalization performance** more reliably.
    - This setup highlights how regularization helps prevent the model from fitting noise or memorizing the training set.

    📌 **In short:**  
    Regularization becomes more important — and more visible — when the model has less data to learn from and more unseen data to generalize to.
    """)


    # Slider for split
    test_size = st.slider("Choose test set proportion", 0.5, 0.9, 0.5, 0.05)

    # Split button
    split_clicked = st.button("🔀 Train-Test Split")
    
    # Placeholder for plot
    split_plot_placeholder = st.empty()

    # Perform split and plot
    if split_clicked and st.session_state.data_generated:
        data = st.session_state.synthetic_data
        x = data["x"].reshape(-1, 1)
        y = data["y"]

        # Do the split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        # Store in session for use in model fitting later
        st.session_state.x_train = x_train
        st.session_state.y_train = y_train
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_train, y_train, 'o', ms=5, mec='k', alpha=0.6, label="Train",)
        ax.plot(x_test, y_test, 'o', ms=5, mec='k', alpha=0.6, label="Test",)
        ax.set_title("Train vs Test Split")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()

        split_plot_placeholder.pyplot(fig)
        
    st.subheader("⚙️ Ridge Regression Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

    with col2:
        lambda_ = st.slider("Regularization Strength (λ)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    with col3:
        n_iters = st.slider("Number of Iterations", min_value=100, max_value=10000, value=5000, step=100)

    # Create model with chosen parameters
    model = RidgeRegression(learning_rate=learning_rate, lambda_=lambda_, n_iters=n_iters)

    # Access data from session state
    X_train = st.session_state.x_train
    y_train = st.session_state.y_train
    
    if st.button("🚀 Train Ridge Model"):
        chart_placeholder = st.empty()
        loss_placeholder = st.empty()
        progress_bar = st.progress(0)
        log_placeholder = st.empty()
        loss_log = []

        for i, weights, bias, loss in model.fit_stepwise(X_train, y_train):
            # Only update plots and logs every 100 iterations
            if (i + 1) % 100 == 0 or i == model.n_iters:
                # Log message
                log_msg = f"Iteration {i+1}/{model.n_iters}, Loss: {loss:.5f}"
                loss_log.append(log_msg)
                loss_log = loss_log[-10:]  # Keep only last 10
                log_placeholder.markdown("```\n" + "\n".join(loss_log) + "\n```")

                # Plot regression line
                fig, ax = plt.subplots()
                ax.plot(X_train, y_train, 'o', ms=5, mec='k', alpha=0.6, label="Train Data")
                x_vals = np.linspace(X_train.min(), X_train.max(), 100)
                y_vals = weights[0] * x_vals + bias
                ax.plot(x_vals, y_vals, label="Model Prediction")
                ax.set_title(f"Iteration {i}")
                ax.legend()
                chart_placeholder.pyplot(fig)

                # Plot loss history
                fig2, ax2 = plt.subplots()
                ax2.plot(model.loss_history)
                ax2.set_title("Loss Over Time")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Loss")
                loss_placeholder.pyplot(fig2)

            # Progress bar updates every step
            progress_bar.progress(min((i + 1) / model.n_iters, 1.0))

        st.success("✅ Training complete!")
            
if section == "Lasso Regression":
    st.header("Lasso Regression (L1)")
    
if section == "Elastic Net":
    
    st.header("Elastic Net")