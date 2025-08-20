import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.title("The Perceptron")

section = st.selectbox("", [
    "Overview",
    "Mathematical Foundations",
    "Interactive Demo"
])

if section == "Overview":
    st.header("Overview")

    st.markdown("""
    ### Early Foundations  
    McCulloch and Pitts (1943) proposed the first model of an artificial neuron which might mimic the human brain with interconnected neurons and information represented to that connection.
    
    Rosenblatt (1958) extended this idea into the perceptron, motivated by pattern recognition tasks related to perception.  

    ### Structure  
    In the Perceptron, inputs are real numbers multiplied by weights, excitatory or inhibitory, then summed. The result passes through an activation function serving as a threshold. If the sum is greater than zero the output is one, otherwise zero.  
    """)
    
    st.image("assets/images/perceptron.png", caption="Neuron vs Perceptron", use_container_width=True)
    
    st.markdown("""
    ### Intuition  
    The perceptron functions as a linear binary classifier. It can separate inputs into two classes when the boundary between them is linear.  

    ### Training  
    The perceptron updates its weights when it misclassifies an example, following the perceptron learning rule. This is not gradient descent with a differentiable loss like in logistic regression, but a simple update rule that guarantees convergence for linearly separable data.  

    ### Limitations  
    Minsky and Papert (1969) showed that perceptrons cannot learn problems such as XOR that require non-linear boundaries. This discovery contributed to the AI winter of the 1970s.  
    """)
    st.image("assets/images/linearly_separable.png", caption="Linearly Separable vs. Non-Linearly Separable", use_container_width=True)

    st.markdown("""
    ### Revival  
    The popularization of backpropagation in the 1980s allowed multilayer perceptrons to learn non-linear functions, overcoming the limitations of a single-layer perceptron.  

    ### Summary  
    The perceptron is an algorithm for supervised learning of binary classifiers. As a single layer it learns only linearly separable patterns. In multiple layers with nonlinear activations it becomes the foundation of modern neural networks.  
    """)

if section == "Mathematical Foundations":
    st.header("Mathematical Foundations")

    st.markdown(r"""
    ## Linear Model  
    The perceptron starts off as a linear model that computes a weighted sum of its inputs.  

    $$
    f(x) = \sum_i w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b = \mathbf{w}^T \mathbf{x} + b
    $$

    ## Activation Function  
    The function is passed through a threshold function called the Heaviside or unit step function.
    """)
    
    st.image("assets/images/heaviside_function.png", caption="Heaviside Function", use_container_width=True)
    st.markdown(r"""
                
    In the perceptron we take:
    
    $$
    z = w^T + b
    $$
    
    $$
    g(z) =
      \begin{cases}
      \hat{y} = 1 & \text{if } z \geq 0 \\
      \hat{y} = 0 & \text{if } z < 0
      \end{cases}
    $$

    Without the step activation, the perceptron behaves like **Linear Regression** trained with gradient descent, producing continuous outputs instead of binary ones.  
    
    Replacing the hard threshold with a **sigmoid** produces **Logistic Regression**, which can be viewed as a *soft boundary linear classifier* that outputs class probabilities instead of hard labels.  

    The perceptron prediction is therefore  

    $$
    \hat{y} = g(f(x)) = g(\mathbf{w}^T \mathbf{x} + b)
    $$

    ## Perceptron Update Rule  
    
    Weights are first **randomly initialized** in the range $[-1, 1]$.
    
    When a point is misclassified, the perceptron adjusts its weights and bias.  

    $$
    \mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{w}
    $$
    $$
    b \leftarrow b + \Delta b
    $$  

    with updates  

    $$
    \Delta \mathbf{w} = \alpha (y_i - \hat{y}_i)\mathbf{x}_i
    $$
    $$
    \Delta b = \alpha (y_i - \hat{y}_i)
    $$  
    
    where $\alpha$ is a learning rate from $[0, 1]$
    """)

    st.image("assets/images/update_rule.png", caption="Perceptron update cases", use_container_width=True)

    st.markdown(r"""
    ## Aside on Gradient Descent  
    This update looks similar to gradient descent on binary cross-entropy, but it is not exactly the same.  

    - Gradient descent minimizes a smooth loss function (e.g. cross-entropy or squared error).  
    - The perceptron uses a discontinuous step function with its own mistake-driven update rule.  
    - They look alike because both adjust weights in the direction of the error, scaled by a learning rate.  

    The perceptron rule guarantees convergence only when the dataset is linearly separable, whereas logistic regression with gradient descent always produces a probabilistic model.  
    """)

    st.markdown(r"""
    ## Summary  
    Single-layer perceptrons are **linear classifiers** that can only separate linearly separable data.  

    Multi-layer perceptrons extend this idea to perform **non-linear classification**.  

    When we introduce MLPs in the **Feed Forward Networks** section, perceptrons will take **differentiable activation functions** and be trained with **backpropagation**, which uses gradient descent to minimize a loss function such as the MSE or BCE as seen in Linear and Logistic Regression.  
    """)
    
if section == "Interactive Demo":
    st.header("Interactive Demo")

    st.markdown(r"""
    Generate a 2D dataset, train a perceptron, and watch the **decision boundary** and **loss** evolve **epoch by epoch**.
    """)

    # ---- Controls ----
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        seed = st.number_input("Seed", value=42, step=1)
    with c2:
        noise = st.slider("Noise", 0.00, 1.00, 0.25, 0.01)
    with c3:
        lr = st.number_input("Learning rate (α)", value=0.1, step=0.01, format="%.3f")
    with c4:
        max_epochs = st.number_input("Max epochs", value=200, step=10)
    with c5:
        display_every = st.number_input("Display every n epochs", value=10, min_value=1, step=1)

    b1, b2, b3 = st.columns([1,1,1])
    gen_btn = b1.button("Generate Data")
    init_btn = b2.button("Initialize Model")
    train_btn = b3.button("Train")

    # ---- Placeholders ----
    left, right = st.columns([1, 1])  # equal widths
    with left:
        st.subheader("Decision Boundary")
        boundary_ph = st.empty()
    with right:
        st.subheader("Train / Val Loss per Epoch")
        loss_ph = st.empty()
    log_ph = st.empty()

    # ---- Session state ----
    ss = st.session_state
    if "demo" not in ss: ss.demo = None
    if "model" not in ss: ss.model = None
    if "history" not in ss: ss.history = {"epoch":[], "train_loss":[], "val_loss":[]}
    if "gen" not in ss: ss.gen = None
    if "loss_log" not in ss: ss.loss_log = []
    if "bounds" not in ss: ss.bounds = None  # store plot bounds for consistent axes

    # ---- Helpers ----
    def gen_blobs(n=1000, noise=0.25, seed=48):
        rng = np.random.default_rng(int(seed))
        n2 = n // 2

        # Random means for the two classes in [-3, 3] x [-3, 3]
        mean0 = rng.uniform(-3, 3, size=2)
        mean1 = rng.uniform(-3, 3, size=2)

        X0 = rng.normal(mean0, noise, size=(n2, 2))
        X1 = rng.normal(mean1, noise, size=(n - n2, 2))
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(len(X0), dtype=int), np.ones(len(X1), dtype=int)])
        return X, y, mean0, mean1


    def _compute_bounds(X, pad=0.5):
        x_min, x_max = X[:,0].min()-pad, X[:,0].max()+pad
        y_min, y_max = X[:,1].min()-pad, X[:,1].max()+pad
        return (x_min, x_max, y_min, y_max)

    def plot_boundary(X, y, W=None, b=None, bounds=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))  # SAME SIZE
        ax.scatter(X[y==0,0], X[y==0,1], alpha=0.7, label="Class 0", marker="o")
        ax.scatter(X[y==1,0], X[y==1,1], alpha=0.7, label="Class 1", marker="^")

        if bounds is None:
            bounds = _compute_bounds(X)
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if W is not None and b is not None:
            xs = np.linspace(x_min, x_max, 400)
            # draw either vertical or standard line
            if abs(W[1]) < 1e-10:
                if abs(W[0]) > 1e-10:
                    ax.axvline(x=-b/W[0], linestyle="--", label="Boundary")
            else:
                ys_line = -(W[0]*xs + b)/W[1]
                ax.plot(xs, ys_line, linestyle="--", label="Boundary")

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend(loc="best")
        return fig

    def plot_loss(history):
        fig, ax = plt.subplots(figsize=(5, 5))  # SAME SIZE
        ax.plot(history["epoch"], history["train_loss"], label="Train loss")
        if any(v is not None for v in history["val_loss"]):
            import numpy as np
            vals = [np.nan if v is None else v for v in history["val_loss"]]
            ax.plot(history["epoch"], vals, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Misclassification rate")
        ax.legend(loc="best")
        return fig
    
    def evaluation_score(y_pred, y_true):
        y_pred = np.squeeze(np.array(y_pred))
        y_true = np.squeeze(np.array(y_true))
        assert y_pred.shape == y_true.shape, "Prediction and ground truth shape mismatch."

        TP = FN = TN = FP = 0
        for i in range(len(y_pred)):
            pred_label = int(y_pred[i])
            gt_label = int(y_true[i])
            if pred_label == 0:   # change if you use -1/1
                if pred_label == gt_label:
                    TN += 1
                else:
                    FN += 1
            else:
                if pred_label == gt_label:
                    TP += 1
                else:
                    FP += 1

        accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        final_score = 50 * accuracy + 50 * f1
        cm = np.array([[TN, FP], [FN, TP]])
        return accuracy, precision, recall, f1, final_score, cm

    def plot_confusion_matrix_from_preds(y_true, y_pred, labels=(0,1), normalize=False, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        if normalize:
            with np.errstate(all='ignore'):
                cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(5, 5))  # SAME SIZE as other plots
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        fmt = ".2f" if normalize else "d"
        thresh = np.nanmax(cm) / 2.0 if cm.size else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm[i, j] if cm.size else 0
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color="white" if val > thresh else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    # ---- Actions ----
    if gen_btn:
        X, y, m0, m1 = gen_blobs(n=1000, noise=noise, seed=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=int(seed))
        ss.demo = (X_tr, y_tr, X_val, y_val)
        ss.history = {"epoch":[], "train_loss":[], "val_loss":[]}
        ss.loss_log = []
        ss.bounds = _compute_bounds(X_tr, pad=0.5)

        boundary_ph.pyplot(plot_boundary(X_tr, y_tr, bounds=ss.bounds))
        loss_ph.pyplot(plot_loss(ss.history))
        log_ph.code(f"Dataset generated. Means: {m0.round(2)} vs {m1.round(2)}")


    if init_btn and ss.demo is not None:
        from utils.perceptron import Perceptron
        X_tr, y_tr, X_val, y_val = ss.demo
        ss.model = Perceptron(learning_rate=float(lr), n_iters=int(max_epochs), init_range=(-1,1))
        # Explicitly initialize so we can draw a boundary BEFORE training:
        ss.model.initialize(n_features=X_tr.shape[1], seed=int(seed))
        # Prepare generator for training (not advancing yet)
        ss.gen = ss.model.fit(X_tr, y_tr, X_val, y_val, seed=int(seed))
        # Show initial boundary (random weights)
        boundary_ph.pyplot(plot_boundary(X_tr, y_tr, ss.model.weights, ss.model.bias, bounds=ss.bounds))
        loss_ph.pyplot(plot_loss(ss.history))
        log_ph.code("Model initialized (weights ~ U[-1,1]). Initial boundary shown.")

    if train_btn and ss.gen is not None and ss.demo is not None:
        X_tr, y_tr, X_val, y_val = ss.demo
        # Iterate epochs, but only update UI every `display_every`
        for step in ss.gen:
            e = step["epoch"]
            W, b = step["W"], step["b"]
            tl, vl = step["train_loss"], step["val_loss"]

            ss.history["epoch"].append(e)
            ss.history["train_loss"].append(tl)
            ss.history["val_loss"].append(vl)

            if (e % int(display_every) == 0) or (e == ss.history["epoch"][-1]):
                boundary_ph.pyplot(plot_boundary(X_tr, y_tr, W, b, bounds=ss.bounds))
                loss_ph.pyplot(plot_loss(ss.history))
                log_msg = f"Epoch {e:>3} | train_loss={tl:.3f}" + ("" if vl is None else f" | val_loss={vl:.3f}")
                ss.loss_log.append(log_msg)
                log_ph.code("\n".join(ss.loss_log[-10:]))

        # Final refresh (in case the last epoch was not exactly on an interval)
        if len(ss.history["epoch"]) > 0:
            boundary_ph.pyplot(plot_boundary(X_tr, y_tr, ss.model.weights, ss.model.bias, bounds=ss.bounds))
            loss_ph.pyplot(plot_loss(ss.history))
            
    # --- Confusion Matrix (bottom) ---
    st.markdown("---")
    st.subheader("Confusion Matrices & Metrics")

    cm_norm = st.checkbox("Normalize confusion matrices", value=False)

    if ss.demo is not None and ss.model is not None:
        X_tr, y_tr, X_val, y_val = ss.demo
        y_tr_bin = (y_tr > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)

        # Predictions with current model
        y_tr_pred = ss.model.predict(X_tr)
        y_val_pred = ss.model.predict(X_val)

        # Top row: Train CM + metrics | Val CM + metrics
        cm_cols = st.columns([1, 1])

        # ---- Train ----
        with cm_cols[0]:
            st.caption("**Train**")
            st.pyplot(plot_confusion_matrix_from_preds(
                y_tr_bin, y_tr_pred, labels=(0,1), normalize=cm_norm, title="Train Confusion Matrix"
            ))
            acc, prec, rec, f1, final_score, _ = evaluation_score(y_tr_pred, y_tr_bin)
            st.markdown(
                f"**Accuracy:** {acc:.3f}  •  **Precision:** {prec:.3f}  •  **Recall:** {rec:.3f}  •  **F1:** {f1:.3f}  •  **Final Score:** {final_score:.2f}"
            )

        # ---- Validation ----
        with cm_cols[1]:
            st.caption("**Validation**")
            st.pyplot(plot_confusion_matrix_from_preds(
                y_val_bin, y_val_pred, labels=(0,1), normalize=cm_norm, title="Validation Confusion Matrix"
            ))
            acc, prec, rec, f1, final_score, _ = evaluation_score(y_val_pred, y_val_bin)
            st.markdown(
                f"**Accuracy:** {acc:.3f}  •  **Precision:** {prec:.3f}  •  **Recall:** {rec:.3f}  •  **F1:** {f1:.3f}  •  **Final Score:** {final_score:.2f}"
            )

        # Flavor text (concise)
        st.markdown("""
        *Accuracy* measures overall correctness.  
        *Precision* is “of predicted 1s, how many were truly 1?” (low precision ⇒ many false positives).  
        *Recall* is “of actual 1s, how many did we catch?” (low recall ⇒ many false negatives).  
        *F1* balances precision and recall via their harmonic mean.  
        *Final Score* here is a simple weighted blend: **50% Accuracy + 50% F1** to value both overall correctness and class‑1 detection quality.
        """)
    else:
        st.info("Generate data, initialize, and train the perceptron to view confusion matrices and metrics.")
