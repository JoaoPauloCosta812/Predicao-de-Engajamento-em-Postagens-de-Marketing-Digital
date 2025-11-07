# ==========================================================
# 📈 Predição de Engajamento em Postagens
# Projeto EBAC × Semantix — Marketing Digital orientado por dados
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------
# Configurações da página
# ----------------------------------------------------------
st.set_page_config(page_title="Predição de Engajamento", page_icon="📊", layout="wide")
st.markdown("# 📊 Predição de Engajamento em Postagens")
st.caption("Projeto EBAC × Semantix — Marketing Digital orientado por dados")

# ----------------------------------------------------------
# Carregar dados e modelo
# ----------------------------------------------------------
PROC_PATH = Path("data/processed/social_media_clean.csv")

@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)

@st.cache_resource
def train_model(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    num_cols = [c for c in df.select_dtypes(include=['int64','float64']).columns if c != target_col]
    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns]
    X = df[num_cols + cat_cols].copy()

    prep = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model = Pipeline([
        ('prep', prep),
        ('model', RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    model.fit(X, y)
    return model, num_cols, cat_cols

if PROC_PATH.exists():
    df = load_data(PROC_PATH)
else:
    st.error("❌ Arquivo não encontrado em `data/processed/social_media_clean.csv`. Execute o notebook 01.")
    st.stop()

# Define variável alvo
target = 'likes' if 'likes' in df.columns else 'engagement_score'
model, num_cols, cat_cols = train_model(df, target)

# ----------------------------------------------------------
# 🎛️ Sidebar — Parâmetros do usuário
# ----------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros do Post")

num_hashtags = st.sidebar.slider("🔢 Número de hashtags", 0, 20, 5, 1)
caption_length = st.sidebar.slider("✍️ Tamanho da legenda (caracteres)", 0, 500, 120, 5)
media_type = st.sidebar.selectbox("🖼️ Tipo de mídia", ['Imagem', 'Vídeo', 'Carrossel'])
post_hour = st.sidebar.slider("⏰ Hora da postagem", 0, 23, 20, 1)

dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
day_of_week = st.sidebar.selectbox("📅 Dia da semana", dias_semana)

caption_bins = st.sidebar.selectbox("🧾 Faixa da legenda", ['Curta', 'Média', 'Longa', 'Muito longa'])
periodo = st.sidebar.selectbox("🌙 Período do dia", ['Madrugada', 'Manhã', 'Tarde', 'Noite', 'Tarde da noite'])

# ----------------------------------------------------------
# 🔄 Ajustar os valores para correspondência com o modelo
# ----------------------------------------------------------
# Mapeamentos para manter coerência com os nomes originais da base
media_type = media_type.lower()
caption_bins = caption_bins.lower().replace(" ", "_")
periodo = periodo.lower().replace(" ", "_")
day_of_week_idx = dias_semana.index(day_of_week)  # converte nome em índice

# ----------------------------------------------------------
# 🔢 Montar DataFrame de entrada
# ----------------------------------------------------------
exemplo = pd.DataFrame([{
    'num_hashtags': num_hashtags,
    'caption_length': caption_length,
    'media_type': media_type,
    'post_hour': post_hour,
    'day_of_week': day_of_week_idx,
    'caption_bins': caption_bins,
    'periodo': periodo
}])

# ----------------------------------------------------------
# 🧩 Garantir colunas compatíveis com o modelo
# ----------------------------------------------------------
cols_treino = num_cols + cat_cols
for c in cols_treino:
    if c not in exemplo.columns:
        exemplo[c] = 0 if c in num_cols else ""
exemplo = exemplo[cols_treino]

# ----------------------------------------------------------
# 📊 Predição
# ----------------------------------------------------------
pred = float(model.predict(exemplo)[0])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Engajamento Previsto", f"{int(pred):,}".replace(",", "."))
with col2:
    st.metric("🕓 Hora sugerida", f"{post_hour:02d}:00")
with col3:
    st.metric("🏷️ #Hashtags", str(num_hashtags))

st.divider()

# ----------------------------------------------------------
# 🔍 Insights rápidos da base
# ----------------------------------------------------------
st.subheader("📊 Insights da Base de Dados")

colA, colB = st.columns(2)
with colA:
    st.write("**Distribuição de Engajamento (amostra)**")
    if 'likes' in df.columns:
        st.bar_chart(df['likes'].sample(min(200, len(df))))
    else:
        st.bar_chart(df['engagement_score'].sample(min(200, len(df))))

with colB:
    if 'post_hour' in df.columns:
        st.write("**Média de Engajamento por Hora de Postagem**")
        st.line_chart(df.groupby('post_hour')[target].mean())

st.caption("💡 Ajuste parâmetros e observe padrões — hashtags, tipo de mídia e horário influenciam diretamente o engajamento.")
