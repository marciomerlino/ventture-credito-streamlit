import streamlit as st
import pandas as pd
import requests
from PIL import Image
import os
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"  # Ajuste para o endereço do serviço

# === Configuração inicial ===
st.set_page_config(
    page_title="Simulador de Crédito Ventture",
    page_icon="assets/ventture_icon.png",
    layout="wide"
)

# === Estilo Ventture ===
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    h1,h2,h3 { color:#1E90FF !important; font-family:'Segoe UI',sans-serif !important; }
    p,label,span { color:#1B1B1B !important; font-family:'Segoe UI',sans-serif !important; }
    .stButton>button { background-color:#1E90FF; color:white; border-radius:8px; height:2.5em; width:100%; }
    .stButton>button:hover { background-color:#0078D7; color:white; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.title("💳 Simulador Inteligente de Crédito Ventture")

# === Abas ===
aba_simulador, aba_analise = st.tabs(["🏦 Simulador de Crédito", "📊 Análises do Histórico"])

# === ABA SIMULADOR ===
with aba_simulador:
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(BASE_DIR, "assets", "ventture.jpg")
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=150)
    st.sidebar.header("📋 Dados do Cliente")

    renda = st.sidebar.number_input("Renda Mensal (R$)", 500, 100000, 8000, 500)
    idade = st.sidebar.number_input("Idade", 18, 90, 35, 1)
    valor_credito = st.sidebar.number_input("Valor do Crédito Solicitado (R$)", 1000, 500000, 50000, 1000)
    valor_bem = st.sidebar.number_input("Valor da Garantia (R$)", 1000, 1000000, 80000, 1000)
    liquidez = st.sidebar.selectbox("Liquidez da Garantia", ["baixa","media","alta"])

    if st.sidebar.button("✅ Analisar Crédito"):
        payload = {
            "renda": renda,
            "idade": idade,
            "valor_credito": valor_credito,
            "valor_bem": valor_bem,
            "liquidez": liquidez
        }
        response = requests.post(f"{API_URL}/predict/", json=payload)
        resultado = response.json()

        aprovado = resultado["aprovado"]
        prob = resultado["prob_aprovacao"]
        # NOVO: Extrair o HTML da explicação
        explicacao_html = resultado["explicacao_html"]

        if aprovado == "Sim":
            st.success(f"✅ Crédito Aprovado! Probabilidade: {prob:.2%}")
        else:
            st.error(f"❌ Crédito Negado. Probabilidade: {prob:.2%}")

        st.subheader("📊 Explicação da Decisão (ELI5)")
        st.components.v1.html(explicacao_html, height=400, scrolling=True)

# === ABA ANÁLISES ===
with aba_analise:
    st.subheader("📈 Histórico de Simulações")
    response = requests.get(f"{API_URL}/historico/")
    historico = pd.DataFrame(response.json())

    if historico.empty:
        st.warning("Nenhum histórico disponível ainda.")
    else:
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
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        taxa_por_liquidez = historico.groupby("liquidez")["aprovado"].apply(lambda x: (x=="Sim").mean()*100)
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
            labels=["Até 50k","50–150k","150–300k","300–500k"]
        )
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        historico.groupby('faixa_credito')['prob_aprovacao'].mean().plot(kind='bar', ax=ax3, color="#1E90FF")
        plt.ylabel("Probabilidade média")
        plt.tight_layout()
        st.pyplot(fig3)

        # Exibir tabela com últimas 20 simulações
        st.write("### Histórico Completo")
        st.dataframe(historico.tail(20).sort_values(by="data_hora", ascending=False))

        # Botão para limpar histórico
        if st.button("🗑️ Limpar histórico"):
            requests.post(f"{API_URL}/limpar_historico/")
            st.warning("Histórico apagado!")
