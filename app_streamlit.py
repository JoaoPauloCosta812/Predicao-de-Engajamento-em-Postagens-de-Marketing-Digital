# ==========================================================
# 📊 Predição de Engajamento em Postagens
# Projeto EBAC × Semantix — Marketing Digital orientado por dados
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
# Carregar dados e treinar modelo (cacheados)
# ----------------------------------------------------------
PROC_PATH = Path("data/processed/social_media_clean.csv")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # padroniza algumas colunas que podem vir com maiúsculas/misturas
    if "media_type" in df.columns:
        df["media_type"] = df["media_type"].astype(str).str.lower().str.strip()
        df["media_type"] = df["media_type"].replace({
            "imagem":"image", "vídeo":"video", "video":"video",
            "carrossel":"carousel", "carrosel":"carousel"
        })
    return df

@st.cache_resource
def train_model(df: pd.DataFrame, target_col: str):
    """Treina pipeline com pré-processamento + RandomForest e retorna
    (modelo, lista_numéricas, lista_categóricas)."""
    y = df[target_col]
    num_cols = [c for c in df.select_dtypes(include=["int64","float64"]).columns if c != target_col]
    cat_cols = [c for c in df.select_dtypes(include=["object","category"]).columns]
    X = df[num_cols + cat_cols].copy()

    prep = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    model.fit(X, y)
    return model, num_cols, cat_cols

if not PROC_PATH.exists():
    st.error("❌ Arquivo não encontrado em `data/processed/social_media_clean.csv`. Execute o notebook 01.")
    st.stop()

df = load_data(PROC_PATH)
target = "likes" if "likes" in df.columns else "engagement_score"
model, num_cols, cat_cols = train_model(df, target)

# ----------------------------------------------------------
# 🎛️ Sidebar — Parâmetros do Post (inputs essenciais)
# ----------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros do Post")

num_hashtags = st.sidebar.slider("🔢 Número de hashtags", 0, 20, 5, 1)
caption_length = st.sidebar.slider("✍️ Tamanho da legenda (caracteres)", 0, 500, 120, 5)
media_type_human = st.sidebar.selectbox("🖼️ Tipo de mídia", ["Imagem", "Vídeo", "Carrossel"])
post_hour = st.sidebar.slider("⏰ Hora da postagem", 0, 23, 20, 1)

dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
day_of_week_human = st.sidebar.selectbox("📅 Dia da semana", dias_semana)
day_of_week = dias_semana.index(day_of_week_human)

# Mapeia o select humano para a codificação do dataset
media_type = media_type_human.lower()
media_type = {"imagem":"image", "vídeo":"video", "carrossel":"carousel"}[media_type]

# ----------------------------------------------------------
# 🔄 Features derivadas automáticas (sem pedir ao usuário)
# ----------------------------------------------------------
# caption_bins a partir de caption_length
caption_bins = pd.cut(
    [caption_length],
    bins=[-1, 60, 120, 180, 9999],
    labels=["curta", "média", "longa", "muito_longa"]
)[0]

# periodo a partir de post_hour
periodo = pd.cut(
    [post_hour],
    bins=[-1, 5, 11, 17, 21, 24],
    labels=["madrugada", "manhã", "tarde", "noite", "tarde_da_noite"]  # antigo "late"
)[0]

# ----------------------------------------------------------
# 🔢 DataFrame de entrada para predição
# ----------------------------------------------------------
exemplo = pd.DataFrame([{
    "num_hashtags": num_hashtags,
    "caption_length": caption_length,
    "media_type": media_type,
    "post_hour": post_hour,
    "day_of_week": day_of_week,
    "caption_bins": str(caption_bins),
    "periodo": str(periodo),
}])

# ----------------------------------------------------------
# 🧩 Garantir compatibilidade de colunas com o modelo
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

# ==========================================================
# 🔍 INSIGHTS EXTRAS
# ==========================================================
st.subheader("📊 Insights da Base de Dados")

colA, colB = st.columns(2)

# a) Média por tipo de mídia
with colA:
    st.write("**Média de Engajamento por Tipo de Mídia**")
    if "media_type" in df.columns:
        df_media = df.copy()
        df_media["media_type"] = df_media["media_type"].astype(str).str.lower()
        media_midia = df_media.groupby("media_type")[target].mean().sort_values(ascending=False)
        st.bar_chart(media_midia)
    else:
        st.info("Coluna `media_type` não encontrada na base processada.")

# b) Heatmap por dia × hora
with colB:
    if set(["day_of_week", "post_hour"]).issubset(df.columns):
        st.write("**Mapa de Calor — Engajamento por Dia da Semana × Hora**")
        pivot = df.pivot_table(values=target, index="day_of_week", columns="post_hour", aggfunc="mean")
        # Deixa a ordem do índice 0..6 (Seg..Dom) se vier desordenado
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        st.dataframe(pivot.style.background_gradient(cmap="YlGnBu"), use_container_width=True)
    else:
        st.info("Colunas `day_of_week` e/ou `post_hour` ausentes para o heatmap.")

# c) Correlação com o alvo
st.write("**Correlação com o Alvo (variáveis numéricas)**")
num_df = df.select_dtypes("number").copy()
if target in num_df.columns:
    corr = num_df.corr()[[target]].sort_values(by=target, ascending=False)
    st.bar_chart(corr)
else:
    st.info("Não foi possível calcular a correlação (alvo numérico ausente).")

# d) Top 5 postagens
st.write("**🏆 Top 5 Postagens com Maior Engajamento**")
cols_vis = [c for c in ["media_type", "num_hashtags", "caption_length", "post_hour", "day_of_week", target] if c in df.columns]
if len(cols_vis) >= 2:
    st.dataframe(df.nlargest(5, target)[cols_vis], use_container_width=True)
else:
    st.info("Colunas necessárias não encontradas para exibir o ranking de postagens.")

# ----------------------------------------------------------
# 🔧 Debug opcional
# ----------------------------------------------------------
with st.expander("⚙️ Debug (opcional)"):
    st.write("Colunas esperadas pelo modelo:", cols_treino)
    st.write("Colunas enviadas no exemplo:", list(exemplo.columns))
