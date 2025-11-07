
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Predição de Engajamento", page_icon="📈", layout="wide")

st.markdown("# 📈 Predição de Engajamento em Postagens")
st.caption("Projeto EBAC × Semantix — Marketing Digital orientado por dados")

# =====================
# Carregar dados
# =====================
PROC_PATH = Path("data/processed/social_media_clean.csv")

@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def train_model(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    num_cols = [c for c in df.select_dtypes(include=['int64','float64']).columns if c != target_col]
    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns]
    X = df[num_cols + cat_cols].copy()
    prep = ColumnTransformer([('num','passthrough', num_cols),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
    model = Pipeline([('prep', prep),
                      ('model', RandomForestRegressor(n_estimators=300, random_state=42))])
    model.fit(X, y)
    return model, num_cols, cat_cols

if PROC_PATH.exists():
    df = load_data(PROC_PATH)
else:
    st.error("Arquivo não encontrado em data/processed/social_media_clean.csv — execute o notebook 01.")
    st.stop()

# Target
target = 'likes' if 'likes' in df.columns else 'engagement_score'
model, num_cols, cat_cols = train_model(df, target)

# =====================
# Sidebar — Inputs
# =====================
st.sidebar.header("🎛️ Parâmetros do Post")
num_hashtags = st.sidebar.slider("Número de hashtags", 0, 20, 5, 1)
caption_length = st.sidebar.slider("Tamanho da legenda (caracteres)", 0, 500, 120, 5)
media_type = st.sidebar.selectbox("Tipo de mídia", ['image','video','carousel'])
post_hour = st.sidebar.slider("Hora da postagem", 0, 23, 20, 1)
day_of_week = st.sidebar.selectbox("Dia da semana (0=Seg ... 6=Dom)", list(range(7)))
caption_bins = st.sidebar.selectbox("Faixa da legenda", ['curta','média','longa','muito_longa'])
periodo = st.sidebar.selectbox("Período do dia", ['madrugada','manhã','tarde','noite','late'])

# =====================
# Predição
# =====================
exemplo = pd.DataFrame([{
    'num_hashtags': num_hashtags,
    'caption_length': caption_length,
    'media_type': media_type,
    'post_hour': post_hour,
    'day_of_week': day_of_week,
    'caption_bins': caption_bins,
    'periodo': periodo
}])

pred = float(model.predict(exemplo)[0])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Predição do Engajamento (alvo)", f"{int(pred):,}".replace(",", "."))
with col2:
    st.metric("⏰ Hora sugerida", f"{post_hour:02d}:00")
with col3:
    st.metric("🏷️ #Hashtags", str(num_hashtags))

st.divider()

# =====================
# Insights rápidos
# =====================
st.subheader("📊 Insights da base")
colA, colB = st.columns(2)
with colA:
    st.write("**Distribuição de likes (amostra)**")
    st.bar_chart(df['likes'].sample(min(200, len(df)))) if 'likes' in df.columns else st.bar_chart(df['engagement_score'].sample(min(200, len(df))))
with colB:
    if 'post_hour' in df.columns:
        st.write("**Média por hora de postagem**")
        st.line_chart(df.groupby('post_hour')[target].mean())

st.caption("💡 Use estas pistas para ajustar sua estratégia: horário, tipo de mídia e #hashtags influenciam bastante o engajamento.")
