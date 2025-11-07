import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor



# Configurações da Página
st.set_page_config(page_title="Predição de Engajamento", page_icon="📈", layout="wide")
st.markdown("# 📈 Predição de Engajamento em Postagens de Marketing Digital")
st.caption("Projeto Final EBAC × Semantix — Marketing Digital orientado por dados")



# Carregar Dados e Treinar Modelo
PROC_PATH = Path("data/processed/social_media_clean.csv")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "media_type" in df.columns:
        df["media_type"] = df["media_type"].astype(str).str.lower().str.strip()
        df["media_type"] = df["media_type"].replace({
            "imagem": "image",
            "vídeo": "video",
            "video": "video",
            "carrossel": "carousel",
            "carrosel": "carousel"
        })
    return df

@st.cache_resource
def train_model(df: pd.DataFrame, target_col: str):
    """Treina pipeline RandomForest + pré-processamento"""
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


# Carregar Base
if not PROC_PATH.exists():
    st.error("❌ Arquivo não encontrado em `data/processed/social_media_clean.csv`. Execute o notebook 01.")
    st.stop()

df = load_data(PROC_PATH)
target = "likes" if "likes" in df.columns else "engagement_score"
model, num_cols, cat_cols = train_model(df, target)


# Sidebar — Parâmetros do Post
st.sidebar.header("🎛️ Parâmetros do Post")

num_hashtags = st.sidebar.slider("🔢 Número de hashtags", 0, 20, 5, 1)
caption_length = st.sidebar.slider("✍️ Tamanho da legenda (caracteres)", 0, 500, 120, 5)
media_type_human = st.sidebar.selectbox("🖼️ Tipo de mídia", ["Imagem", "Vídeo", "Carrossel"])
post_hour = st.sidebar.slider("⏰ Hora da postagem", 0, 23, 20, 1)

dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
day_of_week_human = st.sidebar.selectbox("📅 Dia da semana", dias_semana)
day_of_week = dias_semana.index(day_of_week_human)

# Mapeia selects humanos
media_type = {"imagem": "image", "vídeo": "video", "carrossel": "carousel"}[media_type_human.lower()]

# Faixas derivadas automaticamente
caption_bins = pd.cut([caption_length], bins=[-1,60,120,180,9999],
                      labels=["curta","média","longa","muito_longa"])[0]
periodo = pd.cut([post_hour], bins=[-1,5,11,17,21,24],
                 labels=["madrugada","manhã","tarde","noite","tarde_da_noite"])[0]



# DataFrame de entrada para predição
exemplo = pd.DataFrame([{
    "num_hashtags": num_hashtags,
    "caption_length": caption_length,
    "media_type": media_type,
    "post_hour": post_hour,
    "day_of_week": day_of_week,
    "caption_bins": str(caption_bins),
    "periodo": str(periodo),
}])

cols_treino = num_cols + cat_cols
for c in cols_treino:
    if c not in exemplo.columns:
        exemplo[c] = 0 if c in num_cols else ""
exemplo = exemplo[cols_treino]


# Predição do Modelo
pred = float(model.predict(exemplo)[0])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Engajamento Previsto", f"{int(pred):,}".replace(",", "."))
with col2:
    st.metric("🕓 Hora sugerida", f"{post_hour:02d}:00")
with col3:
    st.metric("🏷️ #Hashtags", str(num_hashtags))

st.divider()


# Filtro dinâmico — gráficos interativos
df_filt = df.copy()

# Filtros contextuais (hora, legenda, hashtags)
df_filt = df_filt[
    (df_filt["post_hour"] >= max(0, post_hour - 2)) &
    (df_filt["post_hour"] <= min(23, post_hour + 2)) &
    (df_filt["caption_length"] >= max(0, caption_length - 50)) &
    (df_filt["caption_length"] <= caption_length + 50) &
    (df_filt["num_hashtags"] >= max(0, num_hashtags - 3)) &
    (df_filt["num_hashtags"] <= num_hashtags + 3)
]

# Filtro adicional por tipo de mídia (para gráficos dependentes)
df_filt_tipo = df_filt.copy()
if "media_type" in df_filt_tipo.columns:
    df_filt_tipo = df_filt_tipo[df_filt_tipo["media_type"] == media_type]

st.caption(
    f"📊 Visualizando postagens com características semelhantes "
    f"(hora≈{post_hour}, legenda≈{caption_length} caracteres, hashtags≈{num_hashtags}).\n"
    f"O gráfico de mídia compara categorias sob esse mesmo contexto."
)



# Análise Exploratória Interativa
st.subheader("📈 Análise Exploratória Interativa")

colA, colB = st.columns(2)

# a) Média de engajamento por tipo de mídia (contextual, mas sem filtrar a mídia selecionada)
with colA:
    st.write("**Média de Engajamento por Tipo de Mídia (contexto atual)**")
    if "media_type" in df_filt.columns:
        media_midia = (
            df_filt.groupby("media_type")[target]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(media_midia)
    else:
        st.info("Coluna `media_type` não encontrada na base.")

# b) Heatmap — engajamento por dia × hora (filtrado por mídia)
with colB:
    if set(["day_of_week", "post_hour"]).issubset(df_filt_tipo.columns):
        st.write(f"**Mapa de Calor — Engajamento por Dia × Hora ({media_type_human})**")
        pivot = df_filt_tipo.pivot_table(values=target, index="day_of_week", columns="post_hour", aggfunc="mean")
        dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
        pivot.index = [dias[i] if i < len(dias) else i for i in range(len(pivot.index))]
        st.dataframe(pivot.style.background_gradient(cmap="YlGnBu"), use_container_width=True)
    else:
        st.info("Colunas necessárias não encontradas para o heatmap.")

# c) Correlação com o alvo
st.write("**Correlação com o Alvo (variáveis numéricas)**")
num_df = df_filt_tipo.select_dtypes("number").copy()
if target in num_df.columns:
    corr = num_df.corr()[[target]].sort_values(by=target, ascending=False)
    st.bar_chart(corr)
else:
    st.info("Não foi possível calcular a correlação (alvo numérico ausente).")

# d) Top 5 postagens
st.write(f"**🏆 Top 5 Postagens ({media_type_human}) com Maior Engajamento**")
cols_vis = [c for c in ["media_type","num_hashtags","caption_length","post_hour","day_of_week",target] if c in df_filt_tipo.columns]
if len(cols_vis) >= 2:
    st.dataframe(df_filt_tipo.nlargest(5, target)[cols_vis], use_container_width=True)
else:
    st.info("Colunas necessárias não encontradas para exibir o ranking.")


# Download da amostra filtrada
st.download_button(
    label="📥 Baixar amostra filtrada (CSV)",
    data=df_filt_tipo.to_csv(index=False, encoding="utf-8-sig"),
    file_name="amostra_filtrada.csv",
    mime="text/csv",
    use_container_width=True)
