# Predição de Engajamento em Postagens de Marketing Digital
Projeto de Parceria | **EBAC × Semantix** — Entregável completo (README + Notebooks + Dashboard + Relatórios).

## Objetivo
Construir um modelo de *Machine Learning* capaz de **prever engajamento** (curtidas/comentários/composto) em postagens
de redes sociais a partir de atributos como **tipo de mídia, horário, #hashtags e tamanho da legenda**. O foco é apoiar
decisões de Marketing Digital com evidências.

## Metodologia
1. **Coleta/Tratamento** — leitura de dataset público do Kaggle (ver seção *Dados*), limpeza, *feature engineering* e fallback sintético.
2. **EDA** — exploração visual (distribuições, correlação, impacto por horário/tipo de mídia).
3. **Modelagem** — regressão (baseline), RandomForest (principal) e XGBoost (opcional).
4. **Avaliação** — R², RMSE, comparação de modelos, interpretação de variáveis (importância).
5. **Visualização** — Dashboard **Streamlit** (inputs -> predição + gráficos).
6. **Relatórios** — versões **sintética (4 págs)** e **completa (8–10 págs)** em PDF.

## Dados
- Preferencial: **Kaggle – Social Media Engagement Dataset** (ou similar com métricas de likes/comments/shares, hashtags, tipo de mídia, horário).
- Caminho esperado: `data/raw/social_media_engagement.csv`.
- **Como usar**: Faça o download no Kaggle e salve o arquivo CSV em `data/raw/` com o nome acima.
- **Fallback**: se o arquivo não existir, o código gera um **dataset sintético** para que o projeto rode 100% offline.

> Briefing oficial do projeto EBAC/Semantix disponível no material: critérios e expectativas de entrega, avaliação e formato de feedback.

## Estrutura
```
Predicao_Engajamento_Marketing/
├── README.md
├── notebooks/
│   ├── 01_coleta_tratamento.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   ├── 03_modelagem_machine_learning.ipynb
│   └── 04_visualizacao_streamlit.ipynb
├── data/
│   ├── raw/social_media_engagement.csv        # (coloque aqui o dataset do Kaggle)
│   └── processed/social_media_clean.csv       # (gerado pelo notebook 01)
├── img/
│   ├── correlacao.png
│   ├── feature_importance.png
│   ├── real_vs_pred.png
│   └── wordcloud_hashtags.png
├── app_streamlit.py
├── relatorio_sintetico.pdf
└── relatorio_completo.pdf
```

## Ambiente
- **Execução local**: Jupyter/VSCode (Python 3.9+)
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, wordcloud, xgboost (opcional), streamlit

## Como rodar
1. Crie um *venv* e instale dependências (ou use conda).
2. Baixe o dataset do Kaggle e salve em `data/raw/social_media_engagement.csv`.
3. Execute os notebooks na ordem `01` → `04`.
4. Rode o dashboard:
   ```bash
   streamlit run app_streamlit.py
   ```

## Resultados esperados
- Identificar variáveis mais relevantes (ex.: **hora**, **tipo de mídia**, **#hashtags**).
- Modelo **RandomForestRegressor** com bom ajuste (R² alto, RMSE baixo).
- Dashboard com **predição interativa** e visualizações.

## Licença e créditos
- Dados: conforme licença do dataset no Kaggle.
- Projeto acadêmico para conclusão do curso **EBAC | Semantix**.
