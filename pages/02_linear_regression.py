import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from utils.linear_regression import standardize_df, minmax_scale_df
from utils.linear_regression import LinearRegression, compute_loss_landscape, plot_2d_hyperplane_with_contour

@st.cache_data
def compute_heatmap_data(df):
    df_copy = df.copy()
    df_copy['lat_bin'] = df_copy['Latitude'].round(1)
    df_copy['lon_bin'] = df_copy['Longitude'].round(1)
    heatmap_df = df_copy.groupby(['lat_bin', 'lon_bin'])['MedHouseVal'].mean().reset_index()
    heatmap_df.rename(columns={'lat_bin': 'lat', 'lon_bin': 'lon', 'MedHouseVal': 'value'}, inplace=True)
    return heatmap_df

st.title("Linear Regression")

# Dropdown for section navigation
section = st.selectbox("", [
    "Overview",
    "What is Linear Regression?",
    "Assumptions of Linear Regression",
    "OLS vs Gradient Descent",
    "Mean Squared Error (MSE)",
    "Gradient Descent Algorithm", 
    "Interactive Demo" 
])

# ---------------------------
if section == "Overview":
    st.header("Overview")
    st.markdown("""
    Welcome to a walkthrough of linear regression using **gradient descent**.
    In this page, we'll explore the assumptions of linear regression, how it's different from the closed-form OLS solution,
    and how to compute and minimize the loss function using gradients. 

    An interactive demo is available which performs linear regression with gradient descent on the 1990 California Housing Dataset
    """)

# ---------------------------
elif section == "What is Linear Regression?":
    st.header("What is Linear Regression?")
    st.markdown(r"""
    Linear regression models the relationship between a dependent variable $y$ and one or more independent variables $x_1, x_2, \dots, x_n$
    by fitting a linear function of the form:
    """)
    st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
    st.markdown(r"""
    Here, $\beta_0$ is the intercept, and $\beta_1, \dots, \beta_n$ are the model coefficients.
    
    The goal of linear regression is to find the coefficients that **minimize the difference** between the predicted values $\hat{y}$ and the actual values $y$ —
    typically by minimizing the **mean squared error** between them.
    """)
    
    st.image("assets/images/linear_regression.png", caption="A best-fit line minimizing the vertical distance (error) between predicted values and observed data.", use_container_width=True)

# ---------------------------
elif section == "Assumptions of Linear Regression":
    st.header("Assumptions of Linear Regression")
    st.markdown("For linear regression to produce reliable estimates and valid inference, several key assumptions must hold.")

    st.markdown("#### 1. Linearity")
    st.markdown("This means that the response variable is a linear combination of the predictors and their coefficients:")
    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n")

    st.markdown("**Nonlinear examples**:")
    st.latex(r"y = \beta_0 + \beta_1 x^2")
    st.latex(r"y = \beta_0 + \beta_1 x + \beta_2 x_1 x_2")
    st.latex(r"y = \beta_0 + \beta_1 \sin(x)")
    st.latex(r"y = \beta_0 + \beta_1 e^{x}")

    st.markdown("#### 2. Constant Variance (Homoscedasticity)")
    st.markdown("The **variance of the error terms** should remain constant across all levels of the input variables.")
    st.latex(r"\text{Var}(\varepsilon_i) = \sigma^2 \quad \text{for all } i")
    st.image("assets/images/homoscedasticity.png", caption="Left: Homoscedasticity — Constant error variance.  Right: Heteroskedasticity — Increasing error variance.", use_container_width=True)

    st.markdown("#### 3. Independence of Errors")
    st.markdown("The **errors should be uncorrelated** with each other.")
    st.latex(r"\text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \quad \text{for } i \neq j")

    st.markdown("#### 4. Imperfect Multicollinearity")
    st.markdown("""
    The predictor variables (covariates) should not be **highly correlated** with each other.

    If two or more covariates are strongly related, it becomes difficult for the model to isolate their individual effects.
    This makes coefficient estimates unstable and inflates their standard errors.
    """)
    st.latex(r"\text{Corr}(x_i, x_j) = \frac{\text{Cov}(x_i, x_j)}{\sigma_{x_i} \sigma_{x_j}} \approx 0")
    st.markdown("A correlation close to ±1 indicates strong linear dependence, which violates the assumption and leads to multicollinearity.")

    st.markdown("### 5. Fixed Inputs (Weak Exogeneity)")
    st.markdown("""
    Linear regression assumes that the input values are **measured without error** and are **non-random** in the context of estimation.

    This is also known as **weak exogeneity** — it means the inputs are not influenced by the outcome, and errors in measuring $x$ don't bias the model.
    """)

# ---------------------------
elif section == "OLS vs Gradient Descent":
    st.header("OLS vs Gradient Descent")
    st.markdown(r"""
    In **Ordinary Least Squares (OLS)**, we solve for the optimal parameters analytically using matrix algebra:
    """)
    st.latex(r"\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y")

    st.markdown("This is efficient for small datasets, but requires computing a **matrix inverse**, which has a time complexity of:")
    st.latex(r"\mathcal{O}(n^3)")

    st.markdown(r"""
    where $n$ is the number of features. This can be computationally expensive and memory-intensive for high-dimensional data.
    """)
    
    st.image("assets/images/complexity.png", caption="Time complexity comparison of OLS (closed-form) and Gradient Descent. OLS grows cubically with the number of features.", use_container_width=True)
    
    st.markdown(r"""
    In contrast, **Gradient Descent** is an **iterative optimization algorithm** that updates parameters gradually by moving in the direction of the negative gradient:
    """)
    st.latex(r"\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}")

    st.markdown("""
    The time complexity of gradient descent per iteration is $\mathcal{O}(n)$.  
    If the algorithm runs for $k$ iterations until convergence, the total time complexity becomes:

    $$
    \mathcal{O}(nk)
    $$

    Compared to the $\mathcal{O}(n^3)$ complexity of matrix inversion in OLS, gradient descent is significantly more scalable for high-dimensional data where total compute cost can be controlled based on the number of iterations required to converge.
                
    This scalability makes gradient descent the preferred method for training large-scale models such as deep neural networks, where closed-form solutions are either unavailable or computationally infeasible.
    """)

elif section == "Mean Squared Error (MSE)":
    st.header("Mean Squared Error (MSE)")

    st.markdown(r"""
    In **Ordinary Least Squares (OLS)**, we minimize the **Residual Sum of Squares (RSS)**:

    $$
    \text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    $$

    When using **gradient descent**, we instead minimize the **Mean Squared Error (MSE)**, which normalizes the RSS by the number of data points:
    """)

    st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    st.image("assets/images/residuals.png", caption="Residuals are the vertical differences between predicted values and actual data points. MSE averages their squared values.", use_container_width=True)

    st.markdown(r"""
    We **square the residuals** because simply summing the differences $y_i - \hat{y}_i$ would lead to positive and negative errors **canceling each other out**.

    By squaring them, we ensure that all errors contribute positively to the total loss.

    ---

    To simplify the math during differentiation, we can use a **modified version** of the MSE by including a $ \frac{1}{2} $ scaling factor:
    """)

    st.latex(r"J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    st.markdown(r"""
    This doesn't affect the minimum — it just cancels constants when taking the gradient.

    ---

    ### Gradient Comparison

    **Gradient of original MSE:**

    $$
    \frac{\partial J}{\partial \theta_j} = \frac{2}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i ) x_{ij}
    $$

    **Gradient of modified MSE:**

    $$
    \frac{\partial J}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i ) x_{ij}
    $$

    As you can see, the modified version removes the factor of 2, making the **gradient descent update rule cleaner and more efficient** to compute during optimization.
    """)

elif section == "Gradient Descent Algorithm":
    st.header("Gradient Descent Algorithm")

    st.markdown(r"""
    Gradient descent is an **iterative optimization algorithm** used to minimize the loss function which is the MSE in the case of linear regression.

    The **gradient** tells us the direction of steepest ascent, so we move in the **opposite direction** to minimize the loss.
    """)

    st.image("assets/images/gradient_descent.png", caption="Gradient descent visualized as a path down the loss surface. The axes represent model weights.", use_container_width=True)

    st.markdown(r"""
    ---

    ### Gradient Descent Update Rule

    For each parameter $ \theta_j $, the update rule is:

    $$
    \theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}
    $$

    Where:
    - $ \theta_j $ is the current parameter value  
    - $ \alpha $ is the **learning rate**  
    - $ \frac{\partial J}{\partial \theta_j} $ is the partial derivative of the loss with respect to $ \theta_j $

    ---

    ### Plugging in the Gradient of MSE

    From the previous section, we know that for the **modified MSE**:

    $$
    J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    $$

    the gradient is:

    $$
    \frac{\partial J}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}
    $$

    Substituting this into the update rule gives:

    $$
    \theta_j := \theta_j - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}
    $$

    This update is applied **simultaneously to all parameters** during each iteration of gradient descent.

    ---

    ### Choosing a Learning Rate ($ \alpha $)

    The learning rate controls **how large each update step is**:

    - If $ \alpha $ is **too small**, training is slow and inefficient  
    - If $ \alpha $ is **too large**, the algorithm may **overshoot** or **diverge**

    A well-tuned $ \alpha $ helps the model converge quickly and smoothly. 
    
    In the image below the learning rate is represented by $\gamma$.
    """)

    st.image("assets/images/learning_rate.png", caption="Effects of different learning rate choices: too small converges slowly, too large overshoots or diverges.", use_container_width=True)
    
    st.markdown(r"""
    ---
    
    ### Gradient Descent Algorithm

    1. Initialize weights $ \theta $
    2. **Repeat until convergence:**
        - Compute predictions $ \hat{y} $
        - Compute gradients:
        $$
        \frac{\partial J}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}
        $$
        - Update parameters:
        $$
        \theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}
        $$
    """)
    st.markdown(r"""
    ---

    ### Convergence Criteria

    Gradient descent usually stops when:
    - The **change in loss** between iterations is below a small threshold  
    - The **gradient norm** becomes very small  
    - A fixed number of **maximum iterations** is reached

    In practice, we monitor the **loss over time** to verify that the model is actually improving.
    """)

elif section == "Interactive Demo":
    st.header("Interactive Demo: Linear Regression with Gradient Descent")

    # -----------------------------
    # Initialize state variables
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.df_scaled = None
        st.session_state.feature_names = []
        st.session_state.x_feature = None
        st.session_state.y_feature = "MedHouseVal"
        st.session_state.scaling_method = "Standard"
        st.session_state.learning_rate = 0.01
        st.session_state.n_iterations = 3000
        st.session_state.test_split_ratio = 0.2
        st.session_state.dataset_loaded = False
        st.session_state.model_trained = False
        st.session_state.train_test_split_done = False
        st.session_state.visualize_feature = None
        st.session_state.visualize_split_feature = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.model = None
        st.session_state.scaler_mean = None
        st.session_state.scaler_std = None
        st.session_state.scaler_min = None
        st.session_state.scaler_max = None
        st.session_state.training_loss_history = []
        st.session_state.training_complete = False
        st.session_state.vmin = None
        st.session_state.vmax = None


    # -----------------------------
    st.markdown("### About the Data")

    st.markdown(r"""
    We're using the **California Housing Dataset (1990)**, which includes information about housing prices and related features like median income, average occupancy, and location-based metrics.

    You’ll be able to select different features, run training live, and make custom predictions interactively.
    """)

    if st.button("Import dataset"):
        from utils.linear_regression import load_california_housing

        with st.spinner("Loading dataset..."):
            st.session_state.df = load_california_housing()
            st.session_state.feature_names = [col for col in st.session_state.df.columns if col != "MedHouseVal"]
            st.session_state.dataset_loaded = True
            rows, cols = st.session_state.df.shape
            st.success(f"✅ Dataset imported successfully! ({rows} rows × {cols} columns)")
            


    if st.session_state.dataset_loaded:
        st.dataframe(st.session_state.df)
        
    if st.session_state.dataset_loaded:
        df_map = st.session_state.df.copy()
        df_map["MedHouseVal_dollars"] = df_map["MedHouseVal"] * 100_000
        st.session_state.vmin = df_map["MedHouseVal_dollars"].min()
        st.session_state.vmax = df_map["MedHouseVal_dollars"].max()

        # Flip longitudes if needed (safety check)
        if df_map["Longitude"].mean() > 0:
            df_map["Longitude"] = df_map["Longitude"].abs()

        st.markdown("### 🗺️ Housing Prices in California (Map View)")
        st.markdown("Each point represents a neighborhood from the 1990 CA housing dataset.")

        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            color="MedHouseVal_dollars",
            size_max=10,
            color_continuous_scale="Viridis",
            zoom=5,
            height=600,
            title="California Housing Prices (1990)",
        )
        fig.update_layout(mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    st.markdown(r"### Feature Descriptions")

    st.markdown(r"""
    In this dataset we will be predicting the Median House Value in 1990 based on 8 different features.
    """)

    with st.expander("Show feature descriptions"):
        st.markdown(r"""
        The **California Housing Dataset** includes the following features:

        | Feature | Description |
        |--------|-------------|
        | `MedInc` | Median income in the block group (in tens of thousands of dollars) |
        | `HouseAge` | Median age of the houses in the block group |
        | `AveRooms` | Average number of rooms per household |
        | `AveBedrms` | Average number of bedrooms per household |
        | `Population` | Total population in the block group |
        | `AveOccup` | Average household occupancy (population / households) |
        | `Latitude` | Latitude of the block group (geographic coordinate) |
        | `Longitude` | Longitude of the block group (geographic coordinate) |
        | `MedHouseVal` | **Target variable**: Median house value in the block group (in hundreds of thousands of dollars) |
        """)
        
    st.markdown("### Explore a Feature vs. Housing Prices")

    # Check if dataset is loaded
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please import the dataset first.")
    else:
        feature = st.selectbox(
            "Select a feature to plot against Median House Value:",
            options=[col for col in st.session_state.df.columns if col != "MedHouseVal"]
        )

        # Extract data
        x = st.session_state.df[feature]
        y = st.session_state.df["MedHouseVal"]

        # Plot
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', ms=5, mec='k', alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel("Median House Value")
        ax.set_title(f"{feature} vs. Median House Value")

        st.pyplot(fig)
        
    st.markdown("As you can see, the **linear correlation** for some features is more obvious than others.")

    # -----------------------------
    st.markdown("### ⚖️ Feature Scaling")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please import the dataset first.")
    else:
        df = st.session_state.df

        st.markdown(r"""
        As you likely noticed, the range of values for the different features varies significantly.
        This can negatively affect gradient descent, which is sensitive to the scale of inputs.
        """)

        with st.expander("Show feature statistics"):
            stats_df = df.drop(columns=["MedHouseVal"]).agg(["min", "max", "mean", "std"]).T
            stats_df["range"] = stats_df["max"] - stats_df["min"]
            st.dataframe(stats_df.sort_values("range", ascending=False))

        if st.button("Plot box plot"):
            from utils.linear_regression import plot_box_plot
            fig = plot_box_plot(df)
            st.pyplot(fig)
            st.info("As you can see, Population has the highest range in the 10,000s while Latitude and Longitude are barely visible.")

        st.markdown("### Feature Scaling & Distribution")
        
        st.markdown(r"""
        Before running gradient descent, it’s important to scale features appropriately.
        We can explore the **distribution of feature values** using density plots to understand whether standardization or min-max scaling would help.

        You’ll choose a scaling method below. That version of the DataFrame will be used for the rest of the demo.
        """)

        # Scaling method selection
        scale_method = st.radio("Choose a scaling method:", ["None", "Standardization", "Min-Max Scaling"])

        if scale_method == "None":
            st.markdown(r"""
            **No scaling** means the raw feature values will be used as-is.
            This can lead to poor convergence behavior during gradient descent,
            especially when features are on very different scales.
            """)
            st.session_state.df_scaled = df.copy()
                       
        elif scale_method == "Standardization":
            st.markdown(r"""
            **Standardization** rescales the features to have a mean of 0 and a standard deviation of 1:

            $$
            x_{\text{scaled}} = \frac{x - \mu}{\sigma}
            $$

            This is the most common scaling method for linear regression.
            """)
            
            df_scaled = standardize_df(df)
            st.session_state.df_scaled = df_scaled
                 
        elif scale_method == "Min-Max Scaling":
            st.markdown(r"""
            **Min-Max Scaling** rescales features to lie in the range $[0, 1]$:

            $$
            x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
            $$

            This is useful when feature ranges need to be strictly bounded.
            """)
            
            df_scaled = minmax_scale_df(df)
            st.session_state.df_scaled = df_scaled          
            
        st.session_state.scaling_method = scale_method
            
        # Plot density panel
        if st.button("Plot Density Plot"):
            from utils.linear_regression import standardize_df, minmax_scale_df, plot_density_panel

            if 'df' not in locals():
                st.warning("⚠️ Please import the dataset first.")
            else:
                if scale_method == "Standardization":
                    df_scaled = standardize_df(df)
                elif scale_method == "Min-Max Scaling":
                    df_scaled = minmax_scale_df(df)
                else:
                    df_scaled = df.copy()
                    
                st.session_state.df_scaled = df_scaled

                fig = plot_density_panel(df_scaled)
                st.pyplot(fig)

                # Add commentary based on scale method
                if scale_method == "None":
                    st.info(
                        "As you can see, some features like `Population` and `AveOccup` are heavily skewed, "
                        "while others like `Latitude` and `HouseAge` appear more symmetric."
                    )
                elif scale_method == "Standardization":
                    st.info(
                        "After standardization, most features are centered around zero and follow a roughly bell-shaped curve. "
                        "This improves convergence when using gradient descent."
                    )
                elif scale_method == "Min-Max Scaling":
                    st.info(
                        "With Min-Max scaling, features are compressed to the [0, 1] range. "
                        "Note how this flattens the distributions and reduces the visual impact of outliers."
                    )
        # -----------------------------
        st.markdown(r"""### Collinearity Check: Pairwise Feature Relationships""")

        st.markdown(r"""
        To visually inspect whether features are **highly correlated with each other**, we use a **pair plot**.
        This plot shows scatter plots for all feature pairs and helps identify **multicollinearity**.
        """)

        if 'df' not in locals():
            st.warning("⚠️ Please import the dataset first.")
        else:
            # Apply selected scaling method
            if scale_method == "Standardization":
                df_scale = standardize_df(df)
            elif scale_method == "Min-Max Scaling":
                df_scale = minmax_scale_df(df)
            else:
                df_scale = df.copy()
                
            st.session_state.df_scaled = df_scale 
            
            # Drop target column for pairplot
            selected_features = st.session_state.df.drop(columns=["MedHouseVal"])

            # Optional: limit to top 5–6 features for readability
            st.markdown("Only plotting top 4 features to keep the visualization readable. This may take awhile...")
            top_features = selected_features.iloc[:, :4]

            if st.button("Plot Pair Plot"):
                with st.spinner("Generating pair plot..."):
                    pair_fig = sns.pairplot(top_features)
                    st.pyplot(pair_fig)

                st.info("As you can see, while most features are roughly uncorrelated, if any two features form a strong diagonal or linear pattern, "
                "it may indicate **high collinearity**, which can affect model interpretability and stability.\n\n"
                "**Example:** `AveRooms` and `AveBedrms` often show a very strong linear relationship, "
                "since both describe household composition.")
                
        st.markdown("### Train-Test Split")

        if st.button("Split Data into Train/Test Sets"):
            if st.session_state.df is None:
                st.warning("⚠️ Please import the dataset first.")
            else:
                from utils.linear_regression import standardize_df, minmax_scale_df

                # Determine which scaling to apply
                # scale_method = st.session_state.scaling_method
                # if scale_method == "Standardization":
                #     df_scaled = standardize_df(st.session_state.df)
                # elif scale_method == "Min-Max Scaling":
                #     df_scaled = minmax_scale_df(st.session_state.df)
                # else:
                #     df_scaled = st.session_state.df.copy()

                # st.session_state.df_scaled = df_scaled
                
                df = st.session_state.df
                scaling_method = st.session_state.scaling_method
                
                X_all = df.drop(columns=["MedHouseVal"])
                y_all = df["MedHouseVal"]
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_all, y_all,
                    test_size=st.session_state.test_split_ratio,
                    random_state=42
                )
                
                st.session_state.X_train_raw = X_train_raw
                st.session_state.X_test_raw = X_test_raw

                # Apply scaling based only on training data
                if scaling_method == "Standardization":
                    mean = X_train_raw.mean()
                    std = X_train_raw.std()
                    X_train = (X_train_raw - mean) / std
                    X_test = (X_test_raw - mean) / std
                    st.session_state.scaler_mean = mean
                    st.session_state.scaler_std = std

                elif scaling_method == "Min-Max Scaling":
                    min_val = X_train_raw.min()
                    max_val = X_train_raw.max()
                    X_train = (X_train_raw - min_val) / (max_val - min_val)
                    X_test = (X_test_raw - min_val) / (max_val - min_val)
                    st.session_state.scaler_min = min_val
                    st.session_state.scaler_max = max_val

                else:  # No scaling
                    X_train = X_train_raw.copy()
                    X_test = X_test_raw.copy()

                # Save scaled versions
                st.session_state.X_train = X_train.values
                st.session_state.X_test = X_test.values
                st.session_state.y_train = y_train.values
                st.session_state.y_test = y_test.values
                st.session_state.feature_names = X_train.columns.tolist()
                st.session_state.train_test_split_done = True

                st.success(f"✅ Data split completed — {len(X_train)} training and {len(X_test)} testing samples.")

        # Visualize feature after splitting
        if st.session_state.get("train_test_split_done", False):
            with st.expander("Visualize Train/Test Split"):
                feature_list = st.session_state.feature_names

                if feature_list:
                    selected_feat = st.selectbox(
                        "Select a feature to visualize against Median House Value:",
                        options=feature_list,
                        index=0,
                        key="visualize_split_feature"
                    )

                    if selected_feat in feature_list:
                        x_idx = feature_list.index(selected_feat)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.scatter(
                            st.session_state.X_train[:, x_idx],
                            st.session_state.y_train,
                            c='tab:blue', label="Train", alpha=0.6
                        )
                        ax.scatter(
                            st.session_state.X_test[:, x_idx],
                            st.session_state.y_test,
                            c='tab:orange', label="Test", alpha=0.6
                        )
                        ax.set_xlabel(selected_feat)
                        ax.set_ylabel("MedHouseVal")
                        ax.set_title("Train/Test Feature Distribution")
                        ax.legend()
                        st.pyplot(fig)

                        st.info("As you can see, train and test data are well-shuffled across the feature space.")
                else:
                    st.warning("⚠️ No features selected to visualize.")
        # -----------------------------
        st.markdown("### 📈 Train the Linear Regression Model")
        st.markdown("""Please select the learning rate and number of iterations to train""")

        if not st.session_state.get("train_test_split_done", False):
            st.warning("⚠️ Please split the data into training and testing sets first.")
        else:
            st.session_state.learning_rate = st.slider(
                "Learning Rate", 
                min_value=0.001, 
                max_value=0.1, 
                step=0.001, 
                value=0.010, 
                format="%.3f"
            )
            st.session_state.n_iterations = st.slider("Number of Iterations", min_value=100, max_value=10000, step=100, value=3000)

            feature_pair = st.multiselect(
                "Select **2 features** for 3D hyperplane visualization and 2D loss contour:",
                options=st.session_state.feature_names,
                default=st.session_state.feature_names[:2],
                max_selections=2,
                key="regression_vis_features"
            )

            if len(feature_pair) != 2:
                st.warning("Please select exactly 2 features.")
            else:
                if st.button("Train Model"):
                    st.session_state.model_trained = False
                    st.session_state.trained_weights = None
                    st.session_state.training_loss_history = []

                    all_feature_names = st.session_state.feature_names
                    all_X_train = st.session_state.X_train
                    y_train = st.session_state.y_train

                    # Get indices of selected features for visualization
                    feat_indices = [all_feature_names.index(f) for f in feature_pair]
                    X_vis = all_X_train[:, feat_indices]

                    # Define theta ranges for contour plot based on feature values
                    theta1_range = (-1, 1)
                    theta2_range = (-1, 1)
                    bias = 0.0  # fixed bias for contour plot calculation

                    # Precompute loss landscape for selected feature weights
                    theta1_vals, theta2_vals, loss_vals = compute_loss_landscape(X_vis, y_train, theta1_range, theta2_range, bias)

                    model = LinearRegression(
                        learning_rate=st.session_state.learning_rate,
                        n_iters=st.session_state.n_iterations
                    )

                    loss_log = []
                    chart_placeholder = st.empty()
                    loss_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    weight_path = []
                    
                    for i, weights, bias, loss in model.fit_stepwise(all_X_train, y_train):
                        weight_pair = [weights[feat_indices[0]], weights[feat_indices[1]]]
                        weight_path.append(weight_pair)
                        
                        if (i + 1) % 100 == 0 or i == st.session_state.n_iterations - 1:
                            fig = plot_2d_hyperplane_with_contour(
                                X_vis, y_train,
                                weights[feat_indices], bias,
                                feature_pair,
                                theta1_vals, theta2_vals, loss_vals,
                                weight_trajectory=weight_path
                            )
                            chart_placeholder.pyplot(fig)

                        if (i + 1) % 100 == 0 or i == st.session_state.n_iterations - 1:
                            log_msg = f"Iteration {i+1}/{st.session_state.n_iterations}, MSE: {loss:.5f}"
                            loss_log.append(log_msg)
                            loss_placeholder.code("\n".join(loss_log[-10:]))
                            progress_bar.progress((i + 1) / st.session_state.n_iterations)

                    st.session_state.model_trained = True
                    st.session_state.trained_weights = weights
                    st.session_state.trained_bias = bias
                    st.session_state.training_loss_history = model.loss_history
                    st.success("✅ Training complete!")
        st.markdown("### ✅ Evaluate")

        # --- Plot Loss Curve ---
        st.markdown("#### 📉 Plot Loss Curve")
        if not st.session_state.get("model_trained", False):
            st.warning("⚠️ Train the model first to see the loss curve.")
        else:
            if st.button("Plot Training Loss"):
                fig_loss = plt.figure(figsize=(8, 4))
                plt.plot(st.session_state.training_loss_history)
                plt.xlabel("Iteration")
                plt.ylabel("MSE Loss")
                plt.title("Training Loss Curve")
                plt.grid(True)
                st.pyplot(fig_loss)

        # --- Evaluate on Test Set ---
        st.markdown("### 🧪 Evaluate on Test Set")

        if not st.session_state.get("model_trained", False):
            st.warning("⚠️ Train the model first to evaluate performance.")
        else:
            # Metric selector
            metric_choice = st.selectbox(
                "📊 Explore Metrics:",
                ["MSE", "R² (Coefficient of Determination)", "Reduced Chi-Squared"]
            )

            # Explanations
            if metric_choice == "MSE":
                st.latex(r"""
                \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
                """)
                st.markdown("""
                **Mean Squared Error (MSE)** measures the average squared difference between actual and predicted values.
                - Lower is better.
                - Always ≥ 0, with 0 being a perfect fit.
                """)
            elif metric_choice == "R² (Coefficient of Determination)":
                st.latex(r"""
                R^2 = 1 - \frac{RSS}{TSS}
                """)
                st.latex(r"""RSS = {\sum (y_i - \hat{y}_i)^2}; \quad TSS = {\sum (y_i - \bar{y})^2}""")
                st.latex(r"""""")
                st.markdown(r"""
                **R²** represents the proportion of variance in the output that is predictable from the input features.
                - **1.0**: perfect prediction
                - **0.0**: model does no better than mean ($\bar{y}$)
                - **< 0.0**: model performs worse than a constant mean predictor
                """)
            elif metric_choice == "Reduced Chi-Squared":
                st.latex(r"""
                \chi^2_\nu = \frac{1}{n - p} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
                """)
                st.markdown("""
                **Reduced Chi-Squared** is a normalized version of the residual sum of squares.
                - **≈ 1.0**: Good fit to the data
                - **≫ 1.0**: Poor fit / underfitting
                - **≪ 1.0**: Overfitting or overestimated noise
                """)

            if st.button("✅ Evaluate"):
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                model = LinearRegression()
                model.weights = st.session_state.trained_weights
                model.bias = st.session_state.trained_bias if "trained_bias" in st.session_state else 0

                y_pred = model.predict(X_test)

                # --- MSE ---
                mse = np.mean((y_test - y_pred) ** 2)
                st.write(f"**Test MSE:** {mse:.5f}")
                if mse < 0.5:
                    st.success("✅ Excellent fit — very low average prediction error.")
                elif mse < 1.5:
                    st.info("Reasonable model — some error, but likely capturing the trend.")
                else:
                    st.warning("⚠️ High MSE — the model might be underfitting or missing key relationships.")

                # --- R² ---
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                st.write(f"**R² Score:** {r2:.5f}")
                if r2 > 0.9:
                    st.success("✅ Excellent — model explains most of the variance.")
                elif r2 > 0.6:
                    st.info("ℹDecent — model explains a good portion of the variance.")
                else:
                    st.warning("⚠️ Low R² — model fails to capture a good portion of the variance.")

                # --- Reduced Chi-Squared ---
                n = len(y_test)
                p = len(model.weights) + 1
                chi2_red = ss_res / (n - p)
                st.write(f"**Reduced χ² (Chi-Squared):** {chi2_red:.5f}")
                if abs(chi2_red - 1.0) < 0.3:
                    st.success("✅ Great fit — residuals are consistent with expected variance.")
                elif chi2_red > 2.0:
                    st.warning("⚠️ χ² too high — likely underfitting.")
                elif chi2_red < 0.5:
                    st.info("ℹχ² very low — possibly overfitting or data noise overestimated.")
                elif chi2_red > 0.5 and chi2_red < 1.0:
                    st.info("χ² low — possibly overfitting or data noise overestimated.")
                    
        # --- Visualize Fit ---
        st.markdown("### 📊 Visualize Fit")

        if not st.session_state.get("model_trained", False):
            st.warning("⚠️ Train the model first to visualize the fit.")
        else:
            st.markdown("Select 1 or 2 features to visualize the fitted model:")

            selected_features = st.multiselect(
                "Select features (1 for 2D line, 2 for 3D hyperplane):",
                options=st.session_state.feature_names,
                default=st.session_state.feature_names[:1],
                max_selections=2,
                key="fit_vis_features"
            )

            if len(selected_features) == 1:
                feature = selected_features[0]
                idx = st.session_state.feature_names.index(feature)

                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                X_feat = X_test[:, idx]

                model = LinearRegression()
                model.weights = st.session_state.trained_weights
                model.bias = st.session_state.trained_bias if "trained_bias" in st.session_state else 0

                # Generate line fit using current model weights
                x_vals = np.linspace(X_feat.min(), X_feat.max(), 100)
                y_vals = model.weights[idx] * x_vals + model.bias

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(X_feat, y_test, color='blue', alpha=0.4, label='True Data')
                ax.plot(x_vals, y_vals, color='red', linewidth=2, label='Regression Line')
                ax.set_xlabel(feature)
                ax.set_ylabel("MedHouseVal")
                ax.set_title(f"2D Linear Fit: {feature} → MedHouseVal")
                ax.legend()
                st.pyplot(fig)

            elif len(selected_features) == 2:
                idx1 = st.session_state.feature_names.index(selected_features[0])
                idx2 = st.session_state.feature_names.index(selected_features[1])

                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                X_vis = X_test[:, [idx1, idx2]]

                model = LinearRegression()
                model.weights = st.session_state.trained_weights
                model.bias = st.session_state.trained_bias if "trained_bias" in st.session_state else 0

                # Create meshgrid for hyperplane
                x_vals = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 50)
                y_vals = np.linspace(X_vis[:, 1].min(), X_vis[:, 1].max(), 50)
                x_grid, y_grid = np.meshgrid(x_vals, y_vals)

                z_vals = (
                    model.weights[idx1] * x_grid +
                    model.weights[idx2] * y_grid +
                    model.bias
                )

                # Plotting
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_vis[:, 0], X_vis[:, 1], y_test, alpha=0.3, label="True Data")
                ax.plot_surface(x_grid, y_grid, z_vals, alpha=0.5, color='red')

                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                ax.set_zlabel("MedHouseVal")
                ax.set_title("3D Regression Hyperplane")
                st.pyplot(fig)

        st.markdown("### 🗺️ California Housing Map — Model Predictions")

        # Ensure required data is available
        if "df" not in st.session_state or "trained_weights" not in st.session_state:
            st.warning("⚠️ Please load data and train model first.")
        else:
            df = st.session_state.df.copy()
            feature_names = st.session_state.feature_names
            scaling_method = st.session_state.scaling_method

            # Build X matrix from full dataset
            X_all = df[feature_names].values

            # Scale using training scalers
            if scaling_method == "Standardization":
                mean = st.session_state.scaler_mean[feature_names]
                std = st.session_state.scaler_std[feature_names]
                X_all = (X_all - mean.values) / std.values
            elif scaling_method == "Min-Max Scaling":
                min_val = st.session_state.scaler_min[feature_names]
                max_val = st.session_state.scaler_max[feature_names]
                X_all = (X_all - min_val.values) / (max_val.values - min_val.values)

            # Predict using model
            model = LinearRegression()
            model.weights = st.session_state.trained_weights
            model.bias = st.session_state.trained_bias if "trained_bias" in st.session_state else 0
            preds = model.predict(X_all) * 100000  # rescale to dollar amount

            # Add to DataFrame
            df["PredictedPrice"] = preds

            # Plot on map
            fig = px.scatter_mapbox(
                df,
                lat="Latitude",
                lon="Longitude",
                color="PredictedPrice",
                color_continuous_scale="Viridis",
                range_color=[st.session_state.vmin, st.session_state.vmax],
                hover_data={"PredictedPrice": ":,.0f"},
                zoom=5,
                height=650,
            )

            fig.update_layout(
                mapbox_style="carto-positron",
                title="Predicted Median House Values Across California",
                margin={"r": 0, "t": 30, "l": 0, "b": 0}
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔮 Custom Prediction")

        if not st.session_state.get("model_trained", False):
            st.warning("⚠️ Train the model first before making custom predictions.")
        else:
            st.markdown("Use the sliders to input custom values for prediction:")

            df_orig = st.session_state.df
            feature_names = st.session_state.feature_names
            feature_values = {}

            # Initialize and cache slider defaults if not already done
            if "slider_defaults" not in st.session_state:
                st.session_state.slider_defaults = {}
                for feature in feature_names:
                    st.session_state.slider_defaults[feature] = {
                        "min": float(df_orig[feature].min()),
                        "max": float(df_orig[feature].max()),
                        "default": float(df_orig[feature].mean())
                    }

            # Get scalers aligned with training feature order
            scaling_method = st.session_state.scaling_method
            if scaling_method == "Standardization":
                mean = st.session_state.scaler_mean[feature_names]
                std = st.session_state.scaler_std[feature_names]
            elif scaling_method == "Min-Max Scaling":
                min_val = st.session_state.scaler_min[feature_names]
                max_val = st.session_state.scaler_max[feature_names]

            # Sliders for each feature
            for feature in feature_names:
                defaults = st.session_state.slider_defaults[feature]
                col1, col2 = st.columns([3, 1])
                with col1:
                    val = st.slider(
                        f"{feature}",
                        min_value=defaults["min"],
                        max_value=defaults["max"],
                        value=defaults["default"],
                        step=0.01,
                        key=f"custom_input_{feature}"
                    )
                    feature_values[feature] = val

            # Build input array
            X_input = np.array([feature_values[f] for f in feature_names]).reshape(1, -1)

            # st.markdown("#### 🔎 Input Before Scaling")
            # st.code(str(X_input))

            # Scale input
            if scaling_method == "Standardization":
                X_input_scaled = (X_input - mean.values) / std.values
            elif scaling_method == "Min-Max Scaling":
                X_input_scaled = (X_input - min_val.values) / (max_val.values - min_val.values)
            else:
                X_input_scaled = X_input

            # st.markdown("#### ⚙️ Input After Scaling")
            # st.code(str(X_input_scaled))

            # Predict
            model = LinearRegression()
            model.weights = st.session_state.trained_weights
            model.bias = st.session_state.trained_bias if "trained_bias" in st.session_state else 0

            y_pred_custom = model.predict(X_input_scaled)[0]

            st.success(f"🏡 Predicted Median House Value: **${y_pred_custom * 100000:,.2f}**")
            
