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
# 1. PAGE SETTINGS
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
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        color: #172554;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dbeafe;
    }
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

# Explicit categories make the encoding easy to interpret.
# With drop="first", the reference categories are:
# cut = Fair, color = D, clarity = I1.
CATEGORY_LEVELS = [
    ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    ["D", "E", "F", "G", "H", "I", "J"],
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
]

# =========================================================
# 2. DOWNLOAD DATA
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
    "Explore diamonds, inspect one-hot encoding, "
    "and predict prices using linear regression."
)

try:
    with st.spinner("Loading Diamonds from OpenML..."):
        df = load_data()
except Exception as error:
    st.error("Unable to load the dataset. Check your internet connection.")
    st.exception(error)
    st.stop()

# =========================================================
# 3. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Model settings")
    st.caption("OpenML dataset ID: 42225")
    st.write("**Target:** Price ($)")
    st.write("**Model:** Multiple linear regression")

    with st.form("settings"):
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

        st.form_submit_button(
            "Apply settings",
            use_container_width=True,
        )

    st.info(
        "cut, color, and clarity are converted into "
        "numerical dummy columns using one-hot encoding."
    )

    st.caption(
        "One category per variable is omitted as the reference "
        "because the regression includes an intercept."
    )

# =========================================================
# 4. DATA EXPLORATION
# =========================================================
explore_tab, encoding_tab, model_tab, predict_tab = st.tabs([
    "📊 Data exploration",
    "🔢 One-hot encoding",
    "📈 Model performance",
    "💰 Predict price",
])

with explore_tab:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diamonds", f"{len(df):,}")
    c2.metric("Original predictors", len(FEATURES))
    c3.metric("Average price", f"${df['price'].mean():,.0f}")
    c4.metric("Missing values", int(df.isna().sum().sum()))

    st.markdown("#### Original dataset")
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
        st.markdown("#### Column information")
        column_info = pd.DataFrame({
            "Data type": df.dtypes.astype(str),
            "Missing values": df.isna().sum(),
            "Unique values": df.nunique(),
        })
        st.dataframe(column_info, use_container_width=True)

        st.write(f"**Duplicate rows:** {df.duplicated().sum():,}")

        invalid_dimensions = (
            df[["x", "y", "z"]] <= 0
        ).any(axis=1).sum()

        st.write(
            "**Rows with nonpositive dimensions:** "
            f"{invalid_dimensions:,}"
        )

    st.markdown("#### Distributions")
    left, right = st.columns(2)

    with left:
        numerical_variable = st.selectbox(
            "Numerical variable",
            ["price"] + NUMERIC,
        )

        fig = px.histogram(
            df,
            x=numerical_variable,
            nbins=60,
            title=f"Distribution of {numerical_variable}",
            color_discrete_sequence=["#2563eb"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        categorical_variable = st.selectbox(
            "Categorical variable",
            CATEGORICAL,
        )

        fig = px.box(
            df,
            x=categorical_variable,
            y="price",
            color=categorical_variable,
            category_orders={
                categorical_variable:
                    CATEGORY_LEVELS[
                        CATEGORICAL.index(categorical_variable)
                    ]
            },
            title=f"Price by {categorical_variable}",
            labels={"price": "Price ($)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Predictor versus price")
    predictor = st.selectbox("Choose predictor", NUMERIC)

    chart_sample = df.sample(
        n=min(5000, len(df)),
        random_state=42,
    )

    fig = px.scatter(
        chart_sample,
        x=predictor,
        y="price",
        color="cut",
        opacity=0.4,
        title=f"{predictor} versus price",
        labels={"price": "Price ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scatter plot displays up to 5,000 sampled observations.")

    st.markdown("#### Numerical correlations")
    correlation = df[NUMERIC + ["price"]].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Exploration describes the full original dataset. "
        "All original rows are retained for modelling."
    )

# =========================================================
# 5. PREPROCESSING AND MODEL TRAINING
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

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                categories=CATEGORY_LEVELS,
                drop="first",
                handle_unknown="error",
                sparse_output=False,
                dtype=np.float64,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])

    # Both preprocessing and regression are fitted on training data.
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


with st.spinner("Preparing the encoded regression model..."):
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        train_predictions,
        test_predictions,
    ) = train_model(
        df,
        test_percent / 100,
        int(seed),
    )

fitted_preprocessor = model.named_steps["preprocessor"]
encoded_feature_names = fitted_preprocessor.get_feature_names_out()

# =========================================================
# 6. DISPLAY ONE-HOT ENCODING
# =========================================================
with encoding_tab:
    st.subheader("Categorical predictors converted into numbers")

    st.write(
        "The model receives only numerical predictors. "
        "Each dummy column contains 0 or 1."
    )

    encoding_summary = pd.DataFrame({
        "Original variable": CATEGORICAL,
        "Categories": [len(levels) for levels in CATEGORY_LEVELS],
        "Dummy columns": [
            len(levels) - 1 for levels in CATEGORY_LEVELS
        ],
        "Reference category": [
            levels[0] for levels in CATEGORY_LEVELS
        ],
    })

    st.dataframe(encoding_summary, use_container_width=True)

    st.success(
        f"Model input: {len(NUMERIC)} numerical predictors + "
        f"{len(encoded_feature_names) - len(NUMERIC)} dummy predictors "
        f"= {len(encoded_feature_names)} numerical columns."
    )

    st.markdown("#### Example: encoding the cut variable")
    cut_example = pd.DataFrame({
        "cut": CATEGORY_LEVELS[0],
    })

    for level in CATEGORY_LEVELS[0][1:]:
        cut_example[f"cut_{level}"] = (
            cut_example["cut"] == level
        ).astype(int)

    st.dataframe(cut_example, use_container_width=True)
    st.caption(
        "Fair is the reference category, so all cut dummy columns "
        "are zero for a Fair diamond."
    )

    st.markdown("#### Actual training data: before encoding")
    preview = X_train.head(10)

    st.dataframe(
        preview.reset_index(drop=True),
        use_container_width=True,
    )

    encoded_preview = pd.DataFrame(
        fitted_preprocessor.transform(preview),
        columns=encoded_feature_names,
        index=preview.index,
    )

    st.markdown("#### Actual training data: after encoding")
    st.dataframe(
        encoded_preview.reset_index(drop=True),
        use_container_width=True,
    )

    st.caption(
        "The encoder is fitted on the training split. "
        "The fitted transformation is reused for test data "
        "and prediction inputs."
    )

    # Prepare the full transformed training data only when requested.
    if st.checkbox("Prepare encoded training dataset for download"):
        encoded_train = pd.DataFrame(
            fitted_preprocessor.transform(X_train),
            columns=encoded_feature_names,
            index=X_train.index,
        )
        encoded_train["price"] = y_train

        st.download_button(
            "Download encoded training data",
            data=encoded_train.to_csv(index=False).encode("utf-8"),
            file_name="diamonds_encoded_training_data.csv",
            mime="text/csv",
        )

# =========================================================
# 7. MODEL EVALUATION
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
        f"Testing observations: **{len(X_test):,}** · "
        f"Encoded predictors: **{len(encoded_feature_names)}**"
    )

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{test_metrics['R²']:.4f}")
    c2.metric("Test RMSE", f"${test_metrics['RMSE ($)']:,.2f}")
    c3.metric("Test MAE", f"${test_metrics['MAE ($)']:,.2f}")

    results = pd.DataFrame(
        [train_metrics, test_metrics],
        index=["Training", "Testing"],
    )
    st.dataframe(results.round(4), use_container_width=True)

    prediction_table = pd.DataFrame({
        "Actual price ($)": y_test.to_numpy(),
        "Predicted price ($)": test_predictions,
    })

    prediction_table["Residual ($)"] = (
        prediction_table["Actual price ($)"]
        - prediction_table["Predicted price ($)"]
    )

    sample = prediction_table.sample(
        n=min(5000, len(prediction_table)),
        random_state=42,
    )

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            sample,
            x="Actual price ($)",
            y="Predicted price ($)",
            opacity=0.4,
            title="Actual versus predicted",
        )

        limits = prediction_table[
            ["Actual price ($)", "Predicted price ($)"]
        ]
        lower = float(limits.min().min())
        upper = float(limits.max().max())

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
            sample,
            x="Predicted price ($)",
            y="Residual ($)",
            opacity=0.4,
            title="Residual plot",
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Diagnostic plots display up to 5,000 test observations.")

    with st.expander("View regression coefficients"):
        coefficients = pd.DataFrame({
            "Predictor": encoded_feature_names,
            "Coefficient": model.named_steps["regression"].coef_,
        })

        st.write(
            "**Intercept:**",
            round(float(model.named_steps["regression"].intercept_), 4),
        )
        st.dataframe(coefficients, use_container_width=True)

        st.caption(
            "Dummy coefficients are relative to Fair, D, and I1, "
            "holding other predictors fixed. Coefficients are not "
            "directly comparable as feature importance."
        )

    st.markdown("#### Test predictions")
    st.dataframe(
        prediction_table.head(100).round(2),
        use_container_width=True,
    )

    st.download_button(
        "Download test predictions",
        data=prediction_table.to_csv(index=False).encode("utf-8"),
        file_name="diamond_test_predictions.csv",
        mime="text/csv",
    )

    st.info(
        "The model predicts original dollar prices. "
        "No log transformation is applied. "
        "Ordinary linear regression can produce negative predictions."
    )

# =========================================================
# 8. PREDICT A NEW DIAMOND
# =========================================================
with predict_tab:
    st.subheader("Predict diamond price")
    st.write(
        "Enter the original characteristics. "
        "The app automatically applies the fitted one-hot encoder."
    )

    with st.form("prediction_form"):
        left, middle, right = st.columns(3)

        with left:
            carat = st.number_input(
                "Carat",
                min_value=0.01,
                max_value=20.0,
                value=1.0,
                step=0.01,
            )

            cut = st.selectbox(
                "Cut",
                CATEGORY_LEVELS[0],
                index=4,
            )

            color = st.selectbox(
                "Color",
                CATEGORY_LEVELS[1],
                index=3,
            )

        with middle:
            clarity = st.selectbox(
                "Clarity",
                CATEGORY_LEVELS[2],
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
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "cut": cut,
            "color": color,
            "clarity": clarity,
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
                "Inputs outside the training range: "
                + ", ".join(outside_range)
                + ". Prediction involves extrapolation."
            )

        implied_depth = 200 * z / (x + y)

        if abs(depth - implied_depth) > 2:
            st.warning(
                f"The dimensions imply depth = {implied_depth:.1f}%, "
                f"but the entered depth is {depth:.1f}%. "
                "Check these inputs."
            )

        # Pipeline automatically one-hot encodes the new input.
        predicted_price = float(model.predict(new_diamond)[0])

        if predicted_price < 0:
            st.error(
                f"Predicted price: ${predicted_price:,.2f}. "
                "This negative value is not physically meaningful "
                "and indicates a limitation of the linear model."
            )
        else:
            st.metric(
                "Estimated price",
                f"${predicted_price:,.2f}",
            )

        st.markdown("#### Your original input")
        st.dataframe(new_diamond, use_container_width=True)

        encoded_input = pd.DataFrame(
            fitted_preprocessor.transform(new_diamond),
            columns=encoded_feature_names,
        )

        st.markdown("#### Numerical input received by the model")
        st.dataframe(encoded_input, use_container_width=True)

        st.caption(
            "Prediction uses the same training-split model evaluated "
            "in the performance tab. This is an estimate, "
            "not a certified valuation."
        )

st.divider()
st.caption("Diamonds · OpenML 42225 · One-hot encoding · Linear regression")
