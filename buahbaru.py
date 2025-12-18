import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# KONFIGURASI STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Buah Berbasis Machine Learning",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("fruit_classification_dataset.csv")

df = load_data()

# --------------------------------------------------
# DEFINISI FITUR & LABEL
# --------------------------------------------------
numerical_features = ['size (cm)', 'weight (g)', 'avg_price (₹)']
categorical_features = ['shape', 'color', 'taste']
target = 'fruit_name'

X = df[numerical_features + categorical_features]
y = df[target]

# --------------------------------------------------
# PREPROCESSING & MODEL
# --------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# --------------------------------------------------
# TRAIN TEST SPLIT & TRAINING
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------
st.sidebar.header("🔍 Input Karakteristik Buah")

input_data = {}

for col in numerical_features:
    input_data[col] = st.sidebar.number_input(
        col,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

for col in categorical_features:
    input_data[col] = st.sidebar.selectbox(
        col,
        df[col].unique()
    )

if st.sidebar.button("Prediksi Jenis Buah"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"🍎 Jenis Buah: **{prediction.upper()}**")

# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------
st.title("🍓 Analisis dan Klasifikasi Jenis Buah by Machine Learning🍎")
st.image(
    "gambar_buah_baru.jpg",
    caption="Klasifikasi Buah Menggunakan Machine Learning",
    use_container_width=True
)
st.markdown(
    "Aplikasi ini memprediksi **jenis buah** berdasarkan ukuran, berat, harga, warna, bentuk, dan rasa."
)

tab1, tab2, tab3 = st.tabs(["Dataset", "EDA", "Model & Evaluasi"])

# --------------------------------------------------
# TAB 1: DATASET
# --------------------------------------------------
with tab1:
    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

# --------------------------------------------------
# TAB 2: EDA
# --------------------------------------------------
with tab2:
    st.subheader("Distribusi Jenis Buah")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        y='fruit_name',
        data=df,
        order=df['fruit_name'].value_counts().index,
        ax=ax
    )
    st.pyplot(fig)

# --------------------------------------------------
# TAB 3: MODEL & EVALUASI
# --------------------------------------------------
with tab3:
    st.metric("Akurasi Model", f"{accuracy*100:.2f}%")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.iloc[:-3])
