import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sistem Rekomendasi Tanaman", page_icon="🌾", layout="wide")

st.markdown("""
    <div style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>🌾 SISTEM REKOMENDASI TANAMAN</h1>
        <p style='color: white; margin:0; font-size: 1.2em;'>Berdasarkan Kondisi Tanah dan Iklim</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("Aplikasi ini memprediksi **tanaman terbaik** berdasarkan kondisi tanah (N, P, K, pH) dan iklim (suhu, kelembaban, curah hujan).")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File 'Crop_recommendation.csv' tidak ditemukan.")
        return None

df = load_data()

if df is not None:
    if "label" not in df.columns:
        st.error("Kolom 'label' tidak ditemukan dalam dataset.")
        st.stop()
    
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    X = df.drop(["label", "label_encoded"], axis=1)
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

with st.sidebar:
    st.header("🌱 Masukkan Kondisi Lahan & Iklim")
    
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
    P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
    K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
    temperature = st.slider("Suhu (°C)", 5.0, 45.0, 25.0, 0.1)
    humidity = st.slider("Kelembaban (%)", 10.0, 100.0, 70.0, 0.1)
    ph = st.slider("pH Tanah", 3.0, 10.0, 6.5, 0.1)
    rainfall = st.slider("Curah Hujan (mm)", 20.0, 300.0, 100.0, 0.1)

    predict_btn = st.button("🔍 Prediksi Tanaman", type="primary", use_container_width=True)
    
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.metric("📊 Akurasi Model", f"{accuracy:.1%}")

if predict_btn and df is not None:
    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    input_scaled = scaler.transform(input_data)
    pred = rf_model.predict(input_scaled)
    crop_name = label_encoder.inverse_transform(pred)[0]
    
    probabilities = rf_model.predict_proba(input_scaled)[0]
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[top_3_idx]

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #a3d977, #56b870); 
                    padding: 40px; 
                    border-radius: 20px; 
                    text-align: center; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin-bottom: 30px;'>
            <h2 style='color: white; margin-bottom: 10px;'>🌱 REKOMENDASI TANAMAN 🌱</h2>
            <h1 style='color: #fff; font-size: 3em; text-transform: uppercase;'>{crop_name}</h1>
            <p style='color: #f0f0f0;'>Berdasarkan kondisi tanah dan iklim yang Anda masukkan</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🏆 3 Rekomendasi Teratas")
        cols = st.columns(3)
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
        
        for crop, prob, col, color in zip(top_3_crops, top_3_probs, cols, colors):
            with col:
                st.markdown(f"""
                <div style='background: {color}; 
                            padding: 20px; 
                            border-radius: 15px; 
                            text-align: center; 
                            color: white;'>
                    <h3>{crop}</h3>
                    <h4>{prob:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📊 Contoh Data Tanaman Serupa")
        similar = df[df["label"] == crop_name].sample(min(5, len(df[df["label"] == crop_name])))
        st.dataframe(similar, use_container_width=True)

elif predict_btn and df is None:
    st.error("Tidak dapat melakukan prediksi karena data tidak tersedia.")

with st.expander("📊 Lihat Statistik Dataset & Visualisasi"):
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Statistik", "🌿 Distribusi Tanaman", "📋 Data Sample"])
        
        with tab1:
            st.write("**Statistik Deskriptif:**")
            st.dataframe(df.describe(), use_container_width=True)
            st.write(f"**Jumlah data:** {df.shape[0]}")
            st.write(f"**Jumlah fitur:** {df.shape[1] - 2}")
            st.write(f"**Total jenis tanaman:** {df['label'].nunique()}")
            
        with tab2:
            st.write("**Distribusi Jenis Tanaman:**")
            crop_counts = df["label"].value_counts()
            st.bar_chart(crop_counts)
            
        with tab3:
            st.write("**Sample Data:**")
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Data tidak tersedia untuk ditampilkan.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Sistem Rekomendasi Tanaman | Dibuat oleh Kelompok Jungle</div>", 
    unsafe_allow_html=True
)
