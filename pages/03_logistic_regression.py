import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logistic_regression import LogisticRegression

def plot_true_vs_pred(model, X_full, y_true, selected_feature_names, all_feature_names):
    """
    Plot 2D feature slice with points colored by true labels (solid) 
    and outlined by predicted labels (edgecolor).
    """
    assert len(selected_feature_names) == 2, "Select exactly 2 features."

    if hasattr(X_full, "to_numpy"):
        X_np = X_full.to_numpy()
    else:
        X_np = X_full

    idx1 = all_feature_names.index(selected_feature_names[0])
    idx2 = all_feature_names.index(selected_feature_names[1])
    X_2d = X_np[:, [idx1, idx2]]

    # Predict using the full feature set
    y_pred = model.predict(X_np)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot solid points (true labels)
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_true,
        cmap="coolwarm",
        s=60,
        alpha=0.7,
        edgecolors="none",
        label="True label"
    )

    # Plot same points again but just as outlines (predicted labels)
    ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        facecolors='none',
        edgecolors=plt.cm.coolwarm(y_pred / y_pred.max()),  # Map 0 or 1 to colormap
        linewidths=1.5,
        s=60,
        label="Predicted label (outline)"
    )

    ax.set_xlabel(selected_feature_names[0])
    ax.set_ylabel(selected_feature_names[1])
    ax.set_title("True vs Predicted Labels (Outline = Prediction)")
    return fig

st.title("Logistic Regression")

section = st.selectbox("", [
    "Overview",
    "Statistical Foundations",
    "Binary Cross-Entropy Loss",
    "Gradient Descent",
    "Interactive Demo" 
])

if section == "Overview":
    st.header("Overview")
    st.markdown(r"""
    ### Linear Regression vs. Logistic Regression
    In linear regression, we model **continuous outcomes** based on input features (also called covariates). 
    
    Logistic regression, in contrast, is used for **classification tasks**, where the outcome is **binary**—such as predicting 0 or 1, no or yes, false or true.

    Rather than modeling the outcome directly, logistic regression models how the **covariates affect the log-odds** of an event. 
    
    $$
    \text{log}\left(\frac{p}{p-1}\right) = \beta_0 + \beta_1 X
    $$
    
    By rearranging this equation, we can solve for the probability \( p \) of the outcome occurring. This yields the **sigmoid function**, which smoothly maps the linear combination of features into a probability between 0 and 1.

    $$
    p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
    $$
    
    In this project, we’ll walk through:
    - What logistic regression is and how it works,
    - The statistical foundations behind the method,
    - The **cross-entropy loss** function that we optimize,
    - How optimization builds a decision boundary,
    - And an **interactive demo** where you can explore each step
    """)
    
if section == "Statistical Foundations":
    st.header("Statistical Foundations of Logistic Regression")
    st.markdown(r"""
    In logistic regression, we model binary outcomes using **Bernoulli random variables**, where:

    $$
    Y \sim \text{Bernoulli}(p)
    $$

    For classification tasks with only two outcomes we can assign each outcome binary values where $p$ is the probability of that outcome.
    - $P(Y = 1) = p$
    - $P(Y = 0) = 1 - p$

    ---

    ### From Covariates to Probabilities

    Our input features (covariates) $X$ can range from $-\infty$ to $+\infty$, but probabilities must lie between 0 and 1.

    To connect the two, we define the **odds** of the event $Y = 1$:

    $$
    \text{Odds} = \frac{p}{1 - p}
    $$

    Odds range from 0 to $+\infty$, but we still need a transformation that spans the **entire real line**, including negative values.

    ---

    ### Modeling with Log-Odds

    Logistic regression models the **log-odds** as a linear function of the input and can span the entire real line:

    $$
    \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X
    $$

    This transformation allows us to:
    - Model probabilities using linear functions of covariates
    - Handle inputs from the full real line
    - Maintain valid probability outputs in $[0, 1]$

    ---

    ### Recovering Probabilities: The Sigmoid
    
    We can extract the raw probability of an outcome by manipulating the log-odds equation:

    $$
    \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X
    $$

    **Exponentiate both sides to eliminate the logarithm:**

    $$
    \frac{p}{1 - p} = e^{\beta_0 + \beta_1 X}
    $$

    **Solve for $p$.**

    $$
    p = (1 - p)e^{\beta_0 + \beta_1 X}
    $$

    **Distribute:**

    $$
    p = e^{\beta_0 + \beta_1 X} - p e^{\beta_0 + \beta_1 X}
    $$

    **Bring all $p$ terms to one side:**

    $$
    p + p e^{\beta_0 + \beta_1 X} = e^{\beta_0 + \beta_1 X}
    $$

    **Factor out $p$:**

    $$
    p(1 + e^{\beta_0 + \beta_1 X}) = e^{\beta_0 + \beta_1 X}
    $$

    **Solve for $p$:**

    $$
    p = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}
    $$

    Now divide the numerator and denominator by $e^{\beta_0 + \beta_1 X}$:

    $$
    p = \frac{1}{\frac{1}{e^{\beta_0 + \beta_1 X}} + 1} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
    $$

    ---

    We define this as the **sigmoid function**:

    $$
    \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \beta_1 X
    $$

    The sigmoid function maps any real-valued input $z$ into a valid probability between 0 and 1.
    """)
    
if section == "Binary Cross-Entropy Loss":
    
    st.markdown(r"""
    ### Binary Cross-Entropy Loss

    In logistic regression, we want to find model parameters $\hat{\beta}$ that best match predicted probabilities to observed binary labels. 
    
    - The goal of learning is to estimate the parameter vector $\hat{\beta}$
    - We divide the training data into labels of either 0 or 1
    - For examples labelled "1" we estimate $\hat{\beta}$ such that $\hat{p(X)}$ is as close to 1 as possible
    - For examples labelled "0" we estimate $\hat{\beta}$ such that $1 - \hat{p(X)}$ is as close to 1 as possible

    Since outputs are binary, we model each label using a **Bernoulli distribution**, and optimize the likelihood of the observed data. This leads directly to the **binary cross-entropy (BCE)** loss.
    """)

    st.markdown(r"""
    ### Deriving Binary Cross-Entropy from Maximum Likelihood

    Each label is modeled as:

    $$
    Y_i \sim \text{Bernoulli}(p_i), \quad \text{where } p_i = \hat{y}_i = \sigma(z_i)
    $$

    The likelihood of observing all $n$ samples is:

    $$
    \mathcal{L}(\beta) = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}
    $$

    Taking the **negative log-likelihood**:

    $$
    -\log \mathcal{L}(\beta) = -\sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
    $$

    Dividing by $n$, we get the **binary cross-entropy loss function**:

    $$
    \text{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
    $$
    """)

    st.markdown(r"""
    ### Why Not Use Mean Squared Error?

    While MSE is popular in regression, it is not appropriate for classification. Here's why:

    **1. MSE assumes a Gaussian likelihood**

    - MSE arises from maximum likelihood under:
      $$
      Y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)
      $$
    - But classification data is binary, and follows:
      $$
      Y_i \sim \text{Bernoulli}(\hat{y}_i)
      $$

    **2. MSE + Sigmoid yields a non-convex loss**

    - The MSE loss becomes non-convex when used with sigmoid outputs, leading to unstable gradients and difficult optimization.
    - BCE, in contrast, is convex with respect to $\hat{y}$.

    **3. MSE under-penalizes confident wrong predictions**

    - MSE gives a relatively low loss and gradient even.
    - BCE reacts more strongly to incorrect high-confidence predictions and adjusts the weights more aggressively.
    """)

    st.markdown(r"""
    ### Example: $y = 0$, $\hat{y} = 0.9$

    **MSE:**
    $$
    L_{\text{MSE}} = (y - \hat{y})^2 = (0 - 0.9)^2 = 0.81
    $$

    Gradient:
    $$
    \frac{dL_{\text{MSE}}}{d\hat{y}} = -2(y - \hat{y}) = 1.8
    $$

    **BCE:**
    $$
    L_{\text{BCE}} = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right] = -\log(1 - 0.9) = -\log(0.1) \approx 2.302
    $$

    Gradient:
    $$
    \frac{dL_{\text{BCE}}}{d\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} = \frac{0.9}{0.09} = 10
    $$

    So at $\hat{y} = 0.9$, BCE applies a much stronger correction (gradient = 10) than MSE (gradient = 1.8).
    """)

    y_hat = np.linspace(0.001, 0.999, 200)

    # BCE and MSE for y = 0
    mse_0 = (0 - y_hat) ** 2
    bce_0 = -np.log(1 - y_hat)

    # BCE and MSE for y = 1
    mse_1 = (1 - y_hat) ** 2
    bce_1 = -np.log(y_hat)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y_hat, mse_0, label="MSE (y = 0)", linestyle="--", color="tab:blue")
    ax.plot(y_hat, bce_0, label="BCE (y = 0)", color="tab:blue")
    ax.plot(y_hat, mse_1, label="MSE (y = 1)", linestyle="--", color="tab:orange")
    ax.plot(y_hat, bce_1, label="BCE (y = 1)", color="tab:orange")

    ax.set_title("Binary Cross-Entropy vs MSE Loss")
    ax.set_xlabel("Predicted Probability ($\hat{y}$)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown(r"""
    As seen in the plot above:
    
    - BCE shoots upward near $\hat{y} = 0$ for $y = 1$, and near $\hat{y} = 1$ for $y = 0$, meaning it strongly penalizes confident mistakes.
    - MSE curves are smoother and less sensitive, leading to smaller corrections during training.

    ---
    
    ✅ **Conclusion**: Binary cross-entropy is the natural choice for logistic regression because it:
    
    - Aligns with the Bernoulli likelihood,
    - Produces convex optimization surfaces (in $\hat{y}$),
    - And penalizes confident errors strongly — which is exactly what we want in classification.
    """)
    
if section == "Gradient Descent":
    st.header("Gradient Descent in Logistic Regression")

    st.markdown(r"""
    ### Objective

    We aim to minimize the **binary cross-entropy loss**:

    $$
    L = -\frac{1}{n}\sum_i^n \left[ y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \right]
    $$

    where:

    - $z = w^\top x + b$
    - $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$

    We now derive:

    $$
    \frac{\partial L}{\partial \beta} = \left[ \frac{\partial L}{\partial w}, \frac{\partial L}{\partial b} \right]
    $$

    ---

    ### Step-by-Step Derivation 

    ### Step 1: Differentiate the Sigmoid Function

    Start with:

    $$
    \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

    Differentiate $\hat{y}$ with respect to $z$ using the quotient rule:

    Let:
    - $u = 1$
    - $v = 1 + e^{-z}$

    Then:

    $$
    \frac{\partial \hat{y}}{\partial z} = \frac{d}{dz} \left( \frac{u}{v} \right)
    = \frac{v \cdot \frac{du}{dz} - u \cdot \frac{dv}{dz}}{v^2}
    = \frac{(1 + e^{-z})(0) - 1 \cdot (-e^{-z})}{(1 + e^{-z})^2}
    = \frac{e^{-z}}{(1 + e^{-z})^2}
    $$

    Now multiply numerator and denominator by $e^z$:

    $$
    \frac{e^{-z}}{(1 + e^{-z})^2}
    = \frac{e^{-z} \cdot e^z}{(1 + e^{-z})^2 \cdot e^z}
    = \frac{1}{(1 + e^{-z})^2 \cdot e^z}
    $$
    
    Recall:

    $$
    \hat{y} = \frac{1}{1 + e^{-z}}
    $$

    Expressing in terms of $\hat{y}$:

    $$
    \frac{1}{(1 + e^{-z})^2 \cdot e^z}
    = \frac{1}{1 + e^{-z}} \cdot \frac{1}{e^z + 1} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \hat{y}\cdot \frac{e^{-z}}{1 + e^{-z}}
    $$

    Now simplify the second factor:

    $$
    \frac{e^{-z}}{1 + e^{-z}}
    = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}}
    = 1 - \frac{1}{1 + e^{-z}} = 1 - \hat{y}
    $$

    So:

    $$
    \frac{\partial \hat{y}}{\partial z} = \frac{e^{-z}}{(1 + e^{-z})^2}
    = \left( \frac{1}{1 + e^{-z}} \right) \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) = \hat{y}(1 - \hat{y})
    $$

    Then:

    $$
    \frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})
    $$

    ---

    ### 🔹 Step 2: Chain Rule for Gradient of Loss

    Recall the loss:

    $$
    L = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
    $$

    Take the derivative of $L$ with respect to $\hat{y}$:

    $$
    \frac{\partial L}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right)
    $$

    Now use chain rule to compute:

    $$
    \frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}
    = \left( -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} \right) \cdot \hat{y}(1 - \hat{y})
    $$

    ---

    ### 🔹 Step 3: Simplify $\frac{\partial L}{\partial z}$ Algebraically

    Let’s expand:

    $$
    \frac{\partial L}{\partial z}
    = -y(1 - \hat{y}) + (1 - y)\hat{y}
    $$

    Distribute terms:

    $$
    \frac{\partial L}{\partial z} =  -y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y
    $$

    So we get:

    $$
    \frac{\partial L}{\partial z} = \hat{y} - y
    $$

    ---

    ### 🔹 Step 4: Derivatives of $z = w^\top x + b$

    $$
    \frac{\partial z}{\partial w} = x \quad \frac{\partial z}{\partial b} = 1
    $$  

    ---

    ### 🔹 Step 5: Apply Chain Rule

    **Derivative with respect to $w$:**

    $$
    \frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{y} - y) \cdot x
    $$

    **Derivative with respect to $b$:**

    $$
    \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = (\hat{y} - y) \cdot 1 = \hat{y} - y
    $$

    ---

    ### Final Gradient

    $$
    \frac{\partial L}{\partial \beta}
    = 
    \begin{bmatrix}
    \frac{\partial L}{\partial w} \\
    \frac{\partial L}{\partial b}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \frac{1}{n}\sum_i^n (\hat{y_i} - y_i) x \\
    \frac{1}{n}\sum_i^n \hat{y_i} - y_i
    \end{bmatrix}
    $$

    ---

    ### Gradient Descent Update

    - Weight update:
    $$
    w := w - \alpha (\hat{y} - y)x
    $$

    - Bias update:
    $$
    b := b - \alpha (\hat{y} - y)
    $$

    where $\alpha$ is the learning rate.

    """)


if section == "Interactive Demo":
    st.header("Interactive Demo: Logistic Regression on Breast Cancer Dataset")
    
    st.markdown(r"""
    ### 🎯 Machine Learning and Logistic Regression

    In machine learning, **logistic regression** is used for **binary classification** — problems where the outcome is either one thing or another (e.g., disease vs. no disease).

    It models the **probability** that an input belongs to class 1 using a sigmoid function applied to a linear combination of features.

    We'll now walk through a full implementation of logistic regression **from scratch**, using only NumPy.
    """)
    
    if st.button("Load Breast Cancer Dataset"):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        df = pd.concat([X, y], axis=1)
        st.session_state["X_raw"] = X
        st.session_state["y_raw"] = y
        st.session_state["df"] = df

        st.success("Dataset loaded!")
        st.dataframe(df)
        

    if "df" in st.session_state:
        st.markdown(r"""
    ### 📊 Visualizing the Data

    Before training, it's helpful to **visualize feature relationships** and how well they separate the two classes.

    We’ll use a **pairplot** to examine how a few features correlate with each other and with the target label. This gives us an intuitive sense of which features may be most informative for classification.
    """)
        if st.button("Generate Pairplot"):
            df = st.session_state["df"]

            # Choose a subset of informative features
            selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]

            # Plot
            fig = sns.pairplot(df[selected_features + ["target"]], hue="target", palette="Set1")
            st.pyplot(fig)

            # Explanatory message below the chart
            st.info("You can see a clear visual separation between the classes, especially along features like mean radius and mean area. Logistic regression should perform well here.")

        st.markdown(r"""
                    ### Train Test Split
                    
                    Now we can split the dataset into the training and testing datasets. We choose an 80-20 split here.
                    """)
        if st.button("Split Train/Test"):
            
            from sklearn.model_selection import train_test_split

            X = st.session_state["X_raw"]
            y = st.session_state["y_raw"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.session_state["X_train_raw"] = X_train
            st.session_state["X_test_raw"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            st.success("Train-test split completed.")
            st.write("X_train shape:", X_train.shape)
            st.write("y_train shape:", y_train.shape)
            
        st.markdown(r"""
                    ### Training
                    
                    After Splitting into train and test sets we can now train our model on our training set.
                    """)
        
        if st.button("Start Training"):
            from utils.logistic_regression import LogisticRegression, plot_losses, sigmoid

            X_train = st.session_state["X_train_raw"].to_numpy()
            X_test = st.session_state["X_test_raw"].to_numpy()
            y_train = st.session_state["y_train"].to_numpy()
            y_test = st.session_state["y_test"].to_numpy()

            model = LogisticRegression(lr=0.001, n_iters=8400)
            

            st.session_state.train_losses = []
            st.session_state.val_losses = []

            # UI elements
            progress_bar = st.progress(0)
            loss_placeholder = st.empty()
            plot_placeholder = st.empty()
            loss_log = []

            for step in model.fit(X_train, y_train, X_test, y_test, log_interval=100):
                st.session_state.train_losses.append(step["train_loss"])
                st.session_state.val_losses.append(step["val_loss"])

                # Logging
                msg = (
                    f"Iteration {step['iteration']:>4}/{model.n_iters} | "
                    f"Train Loss: {step['train_loss']:.5f} | "
                    f"Val Loss: {step['val_loss']:.5f}"
                )
                loss_log.append(msg)
                loss_placeholder.code("\n".join(loss_log[-10:]))  # Show last 10 logs

                # Update loss plot
                fig = plot_losses(
                    st.session_state.train_losses,
                    st.session_state.val_losses,
                    log_interval=100
                )
                plot_placeholder.pyplot(fig)

                # Progress bar update
                progress_bar.progress(step["iteration"] / model.n_iters)
            model.feature_names = list(st.session_state["X_raw"].columns)
            st.session_state["model"] = model 
                
        st.markdown(r"""
### 📈 Prediction and Evaluation

We evaluate the performance of our logistic regression model using several metrics:

- **Accuracy**:
  $$
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{TN} + \text{FN}}
  $$

- **Precision** (Positive Predictive Value):
  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$

- **Recall** (Sensitivity or True Positive Rate):
  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

- **F1 Score** (Harmonic Mean of Precision and Recall):
  $$
  \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

Where:
- **TP**: True Positives
- **FP**: False Positives
- **TN**: True Negatives
- **FN**: False Negatives
""")


        if st.button("Make Predictions"):
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            # Retrieve trained model and data
            model = st.session_state["model"]
            X_train = st.session_state["X_train_raw"]
            X_test = st.session_state["X_test_raw"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate (your function)
            def evaluation_score(y_pred, y_true):
                y_pred = np.squeeze(np.array(y_pred))
                y_true = np.squeeze(np.array(y_true))

                assert y_pred.shape == y_true.shape, "Prediction and ground truth shape mismatch."

                TP = FN = TN = FP = 0

                for i in range(len(y_pred)):
                    pred_label = y_pred[i]
                    gt_label = y_true[i]

                    if int(pred_label) == 0:  # <- change this if your model uses -1/1 instead of 0/1
                        if pred_label == gt_label:
                            TN += 1
                        else:
                            FN += 1
                    else:
                        if pred_label == gt_label:
                            TP += 1
                        else:
                            FP += 1

                accuracy = (TP + TN) / (TP + FN + FP + TN)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                final_score = 50 * accuracy + 50 * f1
                cm = np.array([[TN, FP], [FN, TP]])

                return accuracy, precision, recall, f1, final_score, cm

            # Evaluate both sets
            acc_train, prec_train, rec_train, f1_train, final_train, cm_train = evaluation_score(y_train_pred, y_train)
            acc_test, prec_test, rec_test, f1_test, final_test, cm_test = evaluation_score(y_test_pred, y_test)

            # Plot side-by-side confusion matrices
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Training Set")
                fig1, ax1 = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(ax=ax1, cmap="Blues", colorbar=False)
                st.pyplot(fig1)
                st.metric("Accuracy", f"{acc_train:.2%}")
                st.metric("Precision", f"{prec_train:.2%}")
                st.metric("Recall", f"{rec_train:.2%}")
                st.metric("F1 Score", f"{f1_train:.2%}")
                st.metric("Final Score", f"{final_train:.2f}")

            with col2:
                st.markdown("#### Test Set")
                fig2, ax2 = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(ax=ax2, cmap="Blues", colorbar=False)
                st.pyplot(fig2)
                st.metric("Accuracy", f"{acc_test:.2%}")
                st.metric("Precision", f"{prec_test:.2%}")
                st.metric("Recall", f"{rec_test:.2%}")
                st.metric("F1 Score", f"{f1_test:.2%}")
                st.metric("Final Score", f"{final_test:.2f}")

            # Interpretation notes
            st.info("""
            - **True Positives** (Bottom right): Correctly predicted positive cases.
            - **True Negatives** (Top left): Correctly predicted negative cases.
            - **False Positives** (Top right): Incorrectly predicted as positive.
            - **False Negatives** (Bottom left): Missed positive cases.

            A good model should have most predictions on the **diagonal**.
            """)
            
        st.markdown("### 🔍 Visualize Decision Boundary")

        if "model" in st.session_state:
            all_feats = list(st.session_state["X_raw"].columns)
            selected_feats = st.multiselect("Pick 2 features to compare true vs predicted", all_feats, default=all_feats[:2])

            if len(selected_feats) == 2 and st.button("Visualize Prediction Outline"):
                fig = plot_true_vs_pred(
                    model=st.session_state["model"],
                    X_full=st.session_state["X_raw"],
                    y_true=st.session_state["y_raw"],
                    selected_feature_names=selected_feats,
                    all_feature_names=all_feats
                )
                st.pyplot(fig)
            elif len(selected_feats) != 2:
                st.warning("Please select exactly 2 features.")
        else:
            st.info("Train the model before visualizing decision boundaries.")
            

        st.markdown("### 🔍 Custom Prediction")

        if "model" not in st.session_state:
            st.info("Train the model to enable custom prediction.")
        else:
            model = st.session_state["model"]
            X_raw = st.session_state["X_raw"]
            feature_names = X_raw.columns
            means = X_raw.mean()
            stds = X_raw.std()

            st.markdown("Adjust the sliders to create a custom data point:")

            # Create sliders for each feature
            input_vals = []
            for feat in feature_names:
                mean_val = float(means[feat])
                std_val = float(stds[feat])
                slider_min = round(mean_val - 2.5 * std_val, 2)
                slider_max = round(mean_val + 2.5 * std_val, 2)

                val = st.slider(
                    label=feat,
                    min_value=slider_min,
                    max_value=slider_max,
                    value=round(mean_val, 2),
                    step=0.01
                )
                input_vals.append(val)

            input_array = np.array(input_vals).reshape(1, -1)

            # Predict button
            if st.button("Predict Custom Sample"):
                pred = model.predict(input_array)[0]
                pred_proba = model.predict_proba(input_array)[0]

                st.success(f"**Prediction:** {'Disease' if pred == 1 else 'No Disease'}")
                st.markdown(f"**Probability:** {pred_proba:.4f}")
