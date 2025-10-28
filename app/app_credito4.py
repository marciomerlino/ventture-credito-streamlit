import streamlit as st
import pandas as pd
import requests
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np  # Adicionado para lidar com NaN/Inf no histórico

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
    /* Estilo para a mensagem LLM (FORÇA A COR DO TEXTO PARA PRETO DENTRO DO BLOCO) */
    .stMarkdown { 
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        background-color: #f9f9f9; 
    }
    .stMarkdown p, .stMarkdown * {
        color: #000000 !important; /* Força o texto para preto */
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

st.title("💳 Simulador Inteligente de Crédito Ventture")

# === Abas ===
aba_simulador, aba_analise = st.tabs(["🏦 Simulador de Crédito", "📊 Análises do Histórico"])

# === ABA SIMULADOR ===
with aba_simulador:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Tenta carregar a logo (ajustar o caminho conforme sua estrutura de pastas)
    logo_path = os.path.join(BASE_DIR, "assets", "ventture.jpg")
    try:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=150)
    except FileNotFoundError:
        st.sidebar.header("Ventture Logo Placeholder")

    st.sidebar.header("📋 Dados do Cliente")

    renda = st.sidebar.number_input("Renda Mensal (R$)", 500.0, 100000.0, 8000.0, 500.0)
    idade = st.sidebar.number_input("Idade", 18, 90, 35, 1)
    valor_credito = st.sidebar.number_input("Valor do Crédito Solicitado (R$)", 1000.0, 500000.0, 50000.0, 1000.0)
    valor_bem = st.sidebar.number_input("Valor da Garantia (R$)", 1000.0, 1000000.0, 80000.0, 1000.0)
    liquidez = st.sidebar.selectbox("Liquidez da Garantia", ["baixa", "media", "alta"])

    if st.sidebar.button("✅ Analisar Crédito"):
        payload = {
            "renda": renda,
            "idade": idade,
            "valor_credito": valor_credito,
            "valor_bem": valor_bem,
            "liquidez": liquidez
        }

        # O Streamlit está na porta 8501, a API está na 8000.
        # Adicionado um tratamento de erro básico para falhas de conexão ou na API
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()  # Lança HTTPError para status de erro (4xx ou 5xx)
            resultado = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Erro de conexão: O serviço FastAPI (API) não está ativo ou não respondeu. Certifique-se de que ele está rodando na porta 8000.")
            resultado = None
        except requests.exceptions.HTTPError as err:
            st.error(f"❌ Erro HTTP no serviço: {err}")
            resultado = None
        except Exception as e:
            st.error(f"❌ Erro inesperado ao analisar o crédito: {e}")
            resultado = None

        if resultado is not None:
            if "error" in resultado:
                st.error(f"❌ Erro interno do serviço: {resultado['detail']}")
            else:
                # CORREÇÃO: Extrair todas as chaves, incluindo a nova 'mensagem_cliente'
                aprovado = resultado["aprovado"]
                prob = resultado["prob_aprovacao"]
                explicacao_html = resultado["explicacao_html"]
                mensagem_cliente = resultado["mensagem_cliente"]

                if aprovado == "Sim":
                    st.success(f"✅ Crédito Aprovado! Probabilidade: {prob:.2%}")
                else:
                    st.error(f"❌ Crédito Negado. Probabilidade: {prob:.2%}")

                # NOVO: Exibir a mensagem do LLM
                st.subheader("✉️ Mensagem Personalizada ao Cliente")
                st.markdown(mensagem_cliente)

                st.subheader("📊 Explicação da Decisão (ELI5)")
                # CORRIGIDO: usa a variável explicacao_html extraída do resultado
                st.components.v1.html(explicacao_html, height=400, scrolling=True)

            # === ABA ANÁLISES ===
with aba_analise:
    st.subheader("📈 Histórico de Simulações")

    try:
        response = requests.get(f"{API_URL}/historico/")
        response.raise_for_status()
        historico = pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Falha ao carregar o histórico: {e}")
        historico = pd.DataFrame()

    if historico.empty:
        st.warning("Nenhum histórico disponível ainda.")
    else:
        # Lidar com possíveis NaNs/Infs causados pelo histórico e garantir tipos
        historico = historico.replace([np.inf, -np.inf], np.nan).fillna(value={"mensagem_cliente": ""})

        # Converte 'prob_aprovacao' para float, ignorando erros se houver dados mistos
        historico['prob_aprovacao'] = pd.to_numeric(historico['prob_aprovacao'], errors='coerce')

        # KPIs principais
        total = len(historico)
        taxa_aprovacao = (historico['aprovado'].eq("Sim").mean()) * 100 if total > 0 else 0
        media_prob = historico['prob_aprovacao'].mean() * 100 if 'prob_aprovacao' in historico.columns and not \
        historico['prob_aprovacao'].empty else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Simulações", total)
        col2.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
        col3.metric("Probabilidade Média", f"{media_prob:.1f}%")

        # Gráfico 1: Taxa de aprovação por liquidez
        st.write("### Aprovação por nível de liquidez")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
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
            labels=["Até 50k", "50–150k", "150–300k", "300–500k"],
            right=True
        )
        fig3, ax3 = plt.subplots(figsize=(6, 4))

        # Certifica-se que a coluna 'prob_aprovacao' é numérica antes de agrupar
        historico_agrupado = historico.groupby('faixa_credito', observed=True)['prob_aprovacao'].mean()
        historico_agrupado.plot(kind='bar', ax=ax3, color="#1E90FF")

        plt.ylabel("Probabilidade média")
        plt.tight_layout()
        st.pyplot(fig3)

        # Exibir tabela com últimas 20 simulações
        st.write("### Histórico Completo")
        # CORREÇÃO: Exibir a nova coluna 'mensagem_cliente'
        colunas_display = ['data_hora', 'renda', 'valor_credito', 'aprovado', 'prob_aprovacao', 'mensagem_cliente']
        historico_display = historico.tail(20).sort_values(by="data_hora", ascending=False)
        st.dataframe(historico_display[colunas_display])

        # Botão para limpar histórico
        if st.button("🗑️ Limpar histórico"):
            try:
                requests.post(f"{API_URL}/limpar_historico/")
                st.warning("Histórico apagado! Recarregue a página para confirmar.")
            except Exception:
                st.error("Falha ao se conectar com a API para limpar o histórico.")