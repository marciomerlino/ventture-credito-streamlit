# === app_credito.py ===
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import streamlit.components.v1 as components
from PIL import Image

# === Configuração inicial ===
st.set_page_config(
    page_title="Simulador de Crédito Ventture",
    page_icon="assets/ventture_icon.png",  # coloque o ícone aqui
    layout="wide"
)

# === Estilo Ventture Consulting ===
st.markdown("""
    <style>
        /* Fundo geral */
        .stApp {
            background-color: #FFFFFF;
        }
        /* Títulos principais */
        h1, h2, h3 {
            color: #1E90FF !important;
            font-family: 'Segoe UI', sans-serif !important;
        }
        /* Texto normal */
        p, label, span {
            color: #1B1B1B !important;
            font-family: 'Segoe UI', sans-serif !important;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA;
        }
        /* Botões */
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            height: 2.5em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0078D7;
            color: #FFFFFF;
        }
        /* Rodapé */
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
        div.block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Simulador Inteligente de Concessão de Crédito")

# === Carregar modelo e scaler ===
# Caminhos corretos para modelo e scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_MODELO = os.path.join(BASE_DIR, "model", "modelo_credito.pkl")
ARQUIVO_SCALER = os.path.join(BASE_DIR, "model", "scaler.pkl")
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(ARQUIVO_MODELO)
    scaler = joblib.load(ARQUIVO_SCALER)
    return modelo, scaler

modelo, scaler = carregar_modelo()

# === Caminho do arquivo de histórico ===
arquivo_historico = "historico_simulacoes.csv"

# === Definição das abas ===
aba_simulador, aba_analise = st.tabs(["🏦 Simulador de Crédito", "📊 Análises do Histórico"])

# ========================================================
# 🏦 ABA 1 – SIMULADOR
# ========================================================
with aba_simulador:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(BASE_DIR, "assets", "ventture.jpg")
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=150)
    st.sidebar.header("📋 Dados do Cliente")

    renda = st.sidebar.number_input("Renda Mensal (R$)", min_value=500.0, max_value=100000.0, value=8000.0, step=500.0)
    idade = st.sidebar.number_input("Idade", min_value=18, max_value=90, value=35, step=1)
    valor_credito = st.sidebar.number_input("Valor do Crédito Solicitado (R$)", min_value=1000.0, max_value=500000.0, value=50000.0, step=1000.0)
    valor_bem = st.sidebar.number_input("Valor da Garantia (R$)", min_value=1000.0, max_value=1000000.0, value=80000.0, step=1000.0)
    liquidez = st.sidebar.selectbox("Liquidez da Garantia", ["baixa", "media", "alta"])

    # === Criação das features ===
    liquidez_score = {"baixa": 1, "media": 2, "alta": 3}[liquidez]
    relacao_garantia_credito = valor_bem / (valor_credito + 1)
    renda_por_idade = renda / (idade + 1)
    garantia_ponderada = relacao_garantia_credito * liquidez_score

    # Montar dataframe para previsão
    # df_cliente = pd.DataFrame([{
    #    "renda": renda,
    #    "idade": idade,
    #   "valor_credito": valor_credito,
    #    "valor_bem": valor_bem,
    #    "relacao_garantia_credito": relacao_garantia_credito,
    #    "liquidez_score": liquidez_score,
    #    "renda_por_idade": renda_por_idade,
    #    "garantia_ponderada": garantia_ponderada
    #}])
    # Adiciona a pasta src ao sys.path para permitir import
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

    from preprocess import criar_features_cliente

    df_cliente = criar_features_cliente(
        renda_mensal=renda,
        idade=idade,
        valor_solicitado=valor_credito,
        valor_bem=valor_bem,
        liquidez=liquidez
    )

    # === Normalização ===
    X_scaled = scaler.transform(df_cliente)

    # === Previsão ===
    prob = modelo.predict_proba(X_scaled)[0][1]
    aprovado = int(prob >= 0.5)

    st.subheader("🔍 Resultado da Análise")
    if aprovado:
        st.success(f"✅ Crédito Aprovado! Probabilidade de aprovação: **{prob:.2%}**")
    else:
        st.error(f"❌ Crédito Negado. Probabilidade de aprovação: **{prob:.2%}**")

    # === Gravar no histórico ===
    novo_registro = pd.DataFrame([{
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "renda": renda,
        "idade": idade,
        "valor_credito": valor_credito,
        "valor_bem": valor_bem,
        "liquidez": liquidez,
        "prob_aprovacao": round(prob, 4),
        "aprovado": "Sim" if aprovado else "Não"
    }])

    if os.path.exists(arquivo_historico):
        historico = pd.read_csv(arquivo_historico)
        historico = pd.concat([historico, novo_registro], ignore_index=True)
    else:
        historico = novo_registro

    historico.to_csv(arquivo_historico, index=False)

    # === Explicabilidade com SHAP ===
    st.subheader("📊 Explicação da Decisão (ELI5)")

    # Criar permutação de importância baseado no modelo treinado
    perm = PermutationImportance(modelo, random_state=42)
    perm.fit(X_scaled, [aprovado])  # aqui usamos a previsão como referência

    # Gerar explicação em HTML
    html_obj = eli5.show_weights(
        perm,
        feature_names=df_cliente.columns.tolist(),
        top=8  # mostrar as 8 features mais importantes
    )

    # Adicionar estilo CSS para azul
    html_styled = f"""
    <style>
        body {{
            color: #1E90FF;  /* azul DodgerBlue */
            font-family: Arial, sans-serif;
        }}
        table {{
            color: #1E90FF;
        }}
    </style>
    {html_obj.data}
    """

    # Renderizar no Streamlit
    components.html(html_styled, height=400, scrolling=True)



    # Importância das variáveis
    # st.write("### Impacto das variáveis na decisão:")
    # fig2, ax2 = plt.subplots()
    #shap.summary_plot(shap_values[1], df_cliente, plot_type="bar", show=False)
    # st.pyplot(fig2)

    st.caption("Desenvolvido por Ventture Consulting • Modelo: Random Forest + SHAP • © 2025")

# ========================================================
# 📊 ABA 2 – ANÁLISES
# ========================================================
with aba_analise:

    st.subheader("📈 Análises do Histórico de Simulações")

    if not os.path.exists(arquivo_historico):
        st.warning("Nenhum histórico disponível ainda. Faça uma simulação para começar!")
    else:
        historico = pd.read_csv(arquivo_historico)

        # KPIs principais
        total = len(historico)
        taxa_aprovacao = (historico['aprovado'].eq("Sim").mean()) * 100
        media_prob = historico['prob_aprovacao'].mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Simulações", total)
        col2.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
        col3.metric("Probabilidade Média", f"{media_prob:.1f}%")

        # Gráfico 1: Taxa de aprovação por liquidez
        st.write("### Aprovação por nível de liquidez")
        fig1, ax1 = plt.subplots(figsize=(6, 4))  # menor que o original
        taxa_por_liquidez = historico.groupby("liquidez")["aprovado"].apply(lambda x: (x == "Sim").mean() * 100)
        taxa_por_liquidez.plot(kind="bar", ax=ax1, color="#1E90FF")
        plt.ylabel("% de Aprovação")
        plt.xlabel("Liquidez")
        plt.tight_layout()
        st.pyplot(fig1)

        # Gráfico 2: Distribuição da renda
        st.write("### Distribuição das rendas dos clientes")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        historico['renda'].plot(kind="hist", bins=10, ax=ax2, color="#87CEFA")
        plt.xlabel("Renda (R$)")
        plt.tight_layout()
        st.pyplot(fig2)

        # Gráfico 3: Probabilidade média por faixa de crédito
        st.write("### Probabilidade média por faixa de valor de crédito solicitado")
        historico['faixa_credito'] = pd.cut(
            historico['valor_credito'],
            bins=[0, 50000, 150000, 300000, 500000],
            labels=["Até 50k", "50–150k", "150–300k", "300–500k"]
        )
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        historico.groupby('faixa_credito')['prob_aprovacao'].mean().plot(kind='bar', ax=ax3, color="#1E90FF")
        plt.ylabel("Probabilidade média")
        plt.tight_layout()
        st.pyplot(fig3)

        # Exibir tabela
        st.write("### Histórico Completo")
        st.dataframe(historico.tail(20).sort_values(by="data_hora", ascending=False))

        # Botão para limpar histórico
        if st.button("🗑️ Limpar histórico"):
            os.remove(arquivo_historico)
            st.warning("Histórico apagado!")
