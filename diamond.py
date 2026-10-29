import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        color: #172554;
    }
    h1, h2, h3 {
        color: #172554;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dbeafe;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL = ["cut", "color", "clarity"]
FEATURES = NUMERIC + CATEGORICAL

# =========================================================
# 2. LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    dataset = fetch_openml(
        data_id=42225,
        target_column="price",
        as_frame=True,
    )

    data = dataset.data.copy()
    data["price"] = pd.to_numeric(dataset.target, errors="raise")

    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


st.title("💎 Diamond Price Predictor")
st.write(
    "Explore the Diamonds dataset and predict prices "
    "using multiple linear regression."
)

try:
    with st.spinner("Loading Diamonds from OpenML..."):
        df = load_data()
except Exception as error:
    st.error(
        "Could not download the dataset. Check your internet "
        "connection and restart the app."
    )
    st.exception(error)
    st.stop()

# =========================================================
# 3. SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.title("💎 Model settings")
    st.caption("OpenML dataset ID: 42225")
    st.markdown("**Target:** Price in US dollars")

    with st.form("model_settings"):
        test_percent = st.slider(
            "Testing data (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=100000,
            value=42,
            step=1,
        )

        st.form_submit_button("Apply settings", use_container_width=True)

    st.info(
        "Categorical predictors are one-hot encoded. "
        "Preprocessing is fitted using training data only."
    )

    st.caption(
        "All original rows are retained, including observations "
        "with zero dimensions. These are identified in exploration."
    )

# =========================================================
# 4. DATA EXPLORATION — BEFORE MODEL FITTING
# =========================================================
explore_tab, model_tab, predict_tab = st.tabs([
    "📊 Data exploration",
    "📈 Model performance",
    "💰 Predict price",
])

with explore_tab:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diamonds", f"{len(df):,}")
    c2.metric("Predictors", len(FEATURES))
    c3.metric("Average price", f"${df['price'].mean():,.0f}")
    c4.metric("Missing values", int(df.isna().sum().sum()))

    st.markdown("#### Preview")
    rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(rows), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Numerical summary")
        st.dataframe(
            df.describe().round(2),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Data quality")
        quality = pd.DataFrame({
            "Data type": df.dtypes.astype(str),
            "Missing": df.isna().sum(),
            "Unique values": df.nunique(),
        })
        st.dataframe(quality, use_container_width=True)

        zero_dimensions = int(
            (df[["x", "y", "z"]] <= 0).any(axis=1).sum()
        )

        st.write(f"**Duplicate rows:** {df.duplicated().sum():,}")
        st.write(
            f"**Rows with nonpositive dimensions:** "
            f"{zero_dimensions:,}"
        )

    st.markdown("#### Distributions")
    left, right = st.columns(2)

    with left:
        distribution_feature = st.selectbox(
            "Choose a numerical variable",
            ["price"] + NUMERIC,
        )
        fig = px.histogram(
            df,
            x=distribution_feature,
            nbins=60,
            title=f"Distribution of {distribution_feature}",
            color_discrete_sequence=["#2563eb"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        category = st.selectbox(
            "Compare prices by",
            CATEGORICAL,
        )
        fig = px.box(
            df,
            x=category,
            y="price",
            color=category,
            title=f"Price by {category}",
            labels={"price": "Price ($)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Relationship with price")
    relationship_feature = st.selectbox(
        "Choose predictor",
        NUMERIC,
    )

    # Sample only for chart responsiveness.
    chart_sample = df.sample(
        n=min(5000, len(df)),
        random_state=42,
    )

    fig = px.scatter(
        chart_sample,
        x=relationship_feature,
        y="price",
        color="cut",
        opacity=0.45,
        title=f"{relationship_feature} vs price",
        labels={"price": "Price ($)"},
        hover_data=["color", "clarity"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scatter plot uses up to 5,000 sampled rows.")

    st.markdown("#### Numerical correlations")
    correlation = df[NUMERIC + ["price"]].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pearson correlation matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Exploration describes the full dataset. Model metrics "
        "below are calculated on a separate held-out test split."
    )

# =========================================================
# 5. MODEL TRAINING
# =========================================================
@st.cache_resource(show_spinner=False)
def train_model(data, test_size, random_seed):
    X = data[FEATURES].copy()
    y = data["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        train_predictions,
        test_predictions,
    )


with st.spinner("Preparing the linear regression model..."):
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        train_predictions,
        test_predictions,
    ) = train_model(df, test_percent / 100, int(seed))

# =========================================================
# 6. MODEL EVALUATION
# =========================================================
def calculate_metrics(actual, predicted):
    return {
        "R²": r2_score(actual, predicted),
        "RMSE ($)": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE ($)": mean_absolute_error(actual, predicted),
    }


with model_tab:
    st.subheader("Linear regression performance")

    st.write(
        f"Training observations: **{len(X_train):,}** · "
        f"Testing observations: **{len(X_test):,}**"
    )

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{test_metrics['R²']:.4f}")
    c2.metric("Test RMSE", f"${test_metrics['RMSE ($)']:,.2f}")
    c3.metric("Test MAE", f"${test_metrics['MAE ($)']:,.2f}")

    metrics_table = pd.DataFrame(
        [train_metrics, test_metrics],
        index=["Training", "Testing"],
    )
    st.dataframe(metrics_table.round(4), use_container_width=True)

    st.caption(
        "Higher R² and lower RMSE/MAE indicate better performance. "
        "Errors are measured in original US dollars."
    )

    prediction_table = pd.DataFrame({
        "Actual price ($)": y_test.to_numpy(),
        "Predicted price ($)": test_predictions,
    })
    prediction_table["Residual ($)"] = (
        prediction_table["Actual price ($)"]
        - prediction_table["Predicted price ($)"]
    )

    diagnostic_sample = prediction_table.sample(
        n=min(5000, len(prediction_table)),
        random_state=42,
    )

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            diagnostic_sample,
            x="Actual price ($)",
            y="Predicted price ($)",
            opacity=0.4,
            title="Actual vs predicted",
        )

        lower = float(
            prediction_table[
                ["Actual price ($)", "Predicted price ($)"]
            ].min().min()
        )
        upper = float(
            prediction_table[
                ["Actual price ($)", "Predicted price ($)"]
            ].max().max()
        )

        fig.add_shape(
            type="line",
            x0=lower,
            y0=lower,
            x1=upper,
            y1=upper,
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            diagnostic_sample,
            x="Predicted price ($)",
            y="Residual ($)",
            opacity=0.4,
            title="Residuals vs predicted price",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Diagnostic scatter plots use up to 5,000 test rows.")

    st.markdown("#### Test predictions")
    st.dataframe(
        prediction_table.head(100).round(2),
        use_container_width=True,
    )

    st.download_button(
        "Download all test predictions",
        data=prediction_table.to_csv(index=False).encode("utf-8"),
        file_name="diamond_test_predictions.csv",
        mime="text/csv",
    )

    with st.expander("View regression coefficients"):
        feature_names = (
            model.named_steps["preprocessor"].get_feature_names_out()
        )

        coefficient_table = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.named_steps["regression"].coef_,
        })

        st.write(
            "**Intercept:**",
            round(float(model.named_steps["regression"].intercept_), 4),
        )
        st.dataframe(coefficient_table, use_container_width=True)
        st.caption(
            "Categorical coefficients are relative to the omitted "
            "reference category. Numerical coefficients use different "
            "units and should not be ranked as feature importance."
        )

    st.info(
        "This is ordinary linear regression on original dollar prices. "
        "No logarithmic transformation or outlier removal is applied. "
        "Negative predictions are possible."
    )

# =========================================================
# 7. PREDICT A NEW DIAMOND
# =========================================================
with predict_tab:
    st.subheader("Predict a diamond's price")
    st.write("Enter all nine diamond characteristics.")

    with st.form("diamond_prediction"):
        left, middle, right = st.columns(3)

        with left:
            carat = st.number_input(
                "Carat (weight)",
                min_value=0.01,
                max_value=20.0,
                value=1.0,
                step=0.01,
            )

            cut = st.selectbox(
                "Cut",
                ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                index=4,
            )

            color = st.selectbox(
                "Color",
                ["D", "E", "F", "G", "H", "I", "J"],
                index=3,
            )

        with middle:
            clarity = st.selectbox(
                "Clarity",
                ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
                index=4,
            )

            depth = st.number_input(
                "Depth (%)",
                min_value=1.0,
                max_value=100.0,
                value=61.5,
                step=0.1,
            )

            table = st.number_input(
                "Table (%)",
                min_value=1.0,
                max_value=100.0,
                value=57.0,
                step=0.1,
            )

        with right:
            x = st.number_input(
                "Length x (mm)",
                min_value=0.01,
                max_value=100.0,
                value=6.40,
                step=0.01,
            )

            y = st.number_input(
                "Width y (mm)",
                min_value=0.01,
                max_value=100.0,
                value=6.42,
                step=0.01,
            )

            z = st.number_input(
                "Depth z (mm)",
                min_value=0.01,
                max_value=100.0,
                value=3.94,
                step=0.01,
            )

        submitted = st.form_submit_button(
            "💎 Predict price",
            use_container_width=True,
        )

    if submitted:
        new_diamond = pd.DataFrame([{
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
        }])

        outside_range = [
            column
            for column in NUMERIC
            if not (
                X_train[column].min()
                <= new_diamond[column].iloc[0]
                <= X_train[column].max()
            )
        ]

        if outside_range:
            st.warning(
                "Outside the training range: "
                + ", ".join(outside_range)
                + ". This prediction involves extrapolation."
            )

        # Depth percentage is related to the three dimensions.
        implied_depth = 200 * z / (x + y)
        if abs(depth - implied_depth) > 2.0:
            st.warning(
                f"The dimensions imply a depth of {implied_depth:.1f}%, "
                f"but you entered {depth:.1f}%. Check these inputs."
            )

        prediction = float(model.predict(new_diamond)[0])

        if prediction < 0:
            st.error(
                f"The model predicted ${prediction:,.2f}. "
                "A negative diamond price is not physically meaningful; "
                "this input is poorly represented by the linear model."
            )
        else:
            st.metric("Estimated diamond price", f"${prediction:,.2f}")

        st.dataframe(new_diamond, use_container_width=True)
        st.caption(
            "This is a model estimate, not a certified valuation. "
            "The prediction uses the training-split model evaluated "
            "in the Model performance tab."
        )

st.divider()
st.caption("Diamonds · OpenML 42225 · Multiple Linear Regression")