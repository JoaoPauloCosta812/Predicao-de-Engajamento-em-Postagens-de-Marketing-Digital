import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# Configurações da página
st.set_page_config(page_title="Predição de Engajamento", page_icon="📊", layout="wide")
st.markdown("# 📊 Predição de Engajamento em Postagens")
st.caption("Projeto EBAC × Semantix — Marketing Digital orientado por dados")

PROC_PATH = Path("data/processed/social_media_clean.csv")

@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)
    if "media_type" in df.columns:
        df["media_type"] = df["media_type"].astype(str).str.lower().str.strip()
    return df

@st.cache_resource
def train_model(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    num_cols = [c for c in df.select_dtypes(include=["int64","float64"]).columns if c != target_col]
    cat_cols = [c for c in df.select_dtypes(include=["object","category"]).columns]
    X = df[num_cols + cat_cols].copy()

    prep = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    model.fit(X, y)
    return model, num_cols, cat_cols

if not PROC_PATH.exists():
    st.error("❌ Arquivo não encontrado em `data/processed/social_media_clean.csv`. Execute o notebook 01.")
    st.stop()

df = load_data(PROC_PATH)
target = "likes" if "likes" in df.columns else "engagement_score"
model, num_cols, cat_cols = train_model(df, target)


# Sidebar — Inputs
st.sidebar.header("🎛️ Parâmetros do Post")
num_hashtags = st.sidebar.slider("🔢 Número de hashtags", 0, 20, 5, 1)
caption_length = st.sidebar.slider("✍️ Tamanho da legenda (caracteres)", 0, 500, 120, 5)
media_type_human = st.sidebar.selectbox("🖼️ Tipo de mídia", ["Imagem", "Vídeo", "Carrossel"])
post_hour = st.sidebar.slider("⏰ Hora da postagem", 0, 23, 20, 1)

dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
day_of_week_human = st.sidebar.selectbox("📅 Dia da semana", dias_semana)
day_of_week = dias_semana.index(day_of_week_human)

media_type = {"imagem":"image", "vídeo":"video", "carrossel":"carousel"}[media_type_human.lower()]

caption_bins = pd.cut([caption_length], bins=[-1,60,120,180,9999], labels=["curta","média","longa","muito_longa"])[0]
periodo = pd.cut([post_hour], bins=[-1,5,11,17,21,24], labels=["madrugada","manhã","tarde","noite","tarde_da_noite"])[0]

exemplo = pd.DataFrame([{ "num_hashtags": num_hashtags, "caption_length": caption_length, "media_type": media_type,
                          "post_hour": post_hour, "day_of_week": day_of_week, "caption_bins": str(caption_bins), "periodo": str(periodo)}])

cols_treino = num_cols + cat_cols
for c in cols_treino:
    if c not in exemplo.columns:
        exemplo[c] = 0 if c in num_cols else ""
exemplo = exemplo[cols_treino]

pred = float(model.predict(exemplo)[0])

col1, col2, col3 = st.columns(3)
with col1: st.metric("🎯 Engajamento Previsto", f"{int(pred):,}".replace(",", "."))
with col2: st.metric("🕓 Hora sugerida", f"{post_hour:02d}:00")
with col3: st.metric("🏷️ #Hashtags", str(num_hashtags))

st.divider()


# INSIGHTS EXTRAS
st.subheader("📊 Insights da Base de Dados")
colA, colB = st.columns(2)

# 3️Média por tipo de mídia + aviso 1-barra
with colA:
    st.write("**Média de Engajamento por Tipo de Mídia**")
    if "media_type" in df.columns:
        df_media = df.copy()
        df_media["media_type"] = df_media["media_type"].astype(str).str.lower()
        dist_midia = df_media["media_type"].value_counts()
        media_midia = df_media.groupby("media_type")[target].mean().sort_values(ascending=False)
        st.bar_chart(media_midia)
        if dist_midia.nunique() == 1 or len(dist_midia) == 1:
            st.info("⚠️ A base tem apenas um tipo de mídia; o gráfico exibirá uma única barra.")

# Heatmap com nomes de dias
with colB:
    if set(["day_of_week", "post_hour"]).issubset(df.columns):
        st.write("**Mapa de Calor — Engajamento por Dia da Semana × Hora**")
        nomes_dias = {0:"Segunda",1:"Terça",2:"Quarta",3:"Quinta",4:"Sexta",5:"Sábado",6:"Domingo"}
        pivot = df.pivot_table(values=target, index="day_of_week", columns="post_hour", aggfunc="mean")
        pivot = pivot.rename(index=nomes_dias).reindex(index=list(nomes_dias.values()))
        st.dataframe(pivot.style.background_gradient(cmap="YlGnBu"), use_container_width=True)
    else:
        st.info("Colunas `day_of_week` e/ou `post_hour` ausentes para o heatmap.")

# Download de amostra CSV
amostra = df.sample(min(500, len(df)), random_state=42)
st.download_button(
    "⬇️ Baixar amostra (CSV)",
    amostra.to_csv(index=False).encode("utf-8-sig"),
    file_name="amostra_engajamento.csv",
    mime="text/csv"
)

st.caption("💡 Ajuste horário, tipo de mídia e #hashtags para observar impacto no engajamento previsto.")
