# === credito_service.py ===
# === credito_service.py ===
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
# NOVO: Importar eli5 e PermutationImportance
import eli5
from eli5.sklearn import PermutationImportance

# NOVAS IMPORTAÇÕES PARA GEMINI
from google import genai
from google.genai.errors import APIError
import textwrap # Para formatação de prompts
# NOVO: Carregar variáveis do .env
from dotenv import load_dotenv
load_dotenv()

# === Inicialização da API ===
# ... (código existente da API)

# === Inicialização do Cliente Gemini ===
client = None
try:
    if os.getenv("GEMINI_API_KEY"):
        client = genai.Client()
        print("Cliente Gemini inicializado com sucesso.")
    else:
        print("Atenção: GEMINI_API_KEY não configurada. O serviço de LLM não estará ativo.")
except Exception as e:
    print(f"Atenção: Erro ao inicializar o cliente Gemini: {e}")

# === Inicialização da API ===
app = FastAPI(title="Serviço de Crédito Ventture")

# Permitir que o Streamlit acesse o serviço local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Caminhos ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_MODELO = os.path.join(BASE_DIR, "model", "modelo_credito.pkl")
ARQUIVO_SCALER = os.path.join(BASE_DIR, "model", "scaler.pkl")
ARQUIVO_HISTORICO = os.path.join(BASE_DIR, "historico_simulacoes.csv")

# === Carregar modelo e scaler ===
modelo = joblib.load(ARQUIVO_MODELO)
scaler = joblib.load(ARQUIVO_SCALER)


# === Estrutura dos dados recebidos ===
class CreditoRequest(BaseModel):
    renda: float
    idade: int
    valor_credito: float
    valor_bem: float
    liquidez: str


# === Função para criar features ===
def criar_features_cliente(renda, idade, valor_credito, valor_bem, liquidez):
    liquidez_score = {"baixa": 1, "media": 2, "alta": 3}[liquidez]
    relacao_garantia_credito = valor_bem / (valor_credito + 1)
    renda_por_idade = renda / (idade + 1)
    garantia_ponderada = relacao_garantia_credito * liquidez_score

    df = pd.DataFrame([{
        "renda_mensal": renda,
        "idade": idade,
        "valor_solicitado": valor_credito,
        "valor_bem": valor_bem,
        "relacao_garantia_credito": relacao_garantia_credito,
        "liquidez_score": liquidez_score,
        "renda_por_idade": renda_por_idade,
        "garantia_ponderada": garantia_ponderada
    }])

    return df

def gerar_mensagem_aprovacao(data: CreditoRequest, prob: float) -> str:
    """ Gera a mensagem de aprovação consultiva. """
    if not client:
        return f"Crédito Aprovado com {prob:.2%} de probabilidade! Entre em contato para finalizar (LLM Inativo)."

    taxa_simulada = 8.5  # Valor de exemplo
    prazo_simulado = 36  # Valor de exemplo

    system_prompt = """
    Você é um Gerente de Crédito sênior da Ventture. Sua tarefa é criar uma proposta de valor consultiva, clara e que utilize as regras internas.
    O tom deve ser profissional e focado no valor para o cliente. Use a saudação "Prezado Cliente," no início.
    """

    dados_proposta_prompt = textwrap.dedent(f"""
    INFORMAÇÕES DA ANÁLISE:
    - Renda Mensal: R$ {data.renda:,.2f}
    - Idade: {data.idade} anos
    - Valor Solicitado: R$ {data.valor_credito:,.2f}
    - Garantia (Bem): R$ {data.valor_bem:,.2f} (Liquidez: {data.liquidez.upper()})
    - Probabilidade de Aprovação: {prob:.2%}
    - Condições Sugeridas (simulação): Taxa {taxa_simulada:.2f}% a.a. e Prazo {prazo_simulado} meses.

    Gere a proposta final de valor, focando nos benefícios da sua garantia para o acesso ao crédito e mencione que a próxima etapa é o envio de documentação.
    """)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=dados_proposta_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4
            )
        )
        return response.text.strip()
    except Exception:
        return f"Erro: Não foi possível gerar a mensagem de aprovação personalizada."


def gerar_mensagem_negacao(data: CreditoRequest, prob: float) -> str:
    """ Gera a mensagem de negativa empática. """
    if not client:
        return f"Crédito Negado com {prob:.2%} de probabilidade. Não foi possível atender a sua solicitação neste momento (LLM Inativo)."

    # Criando um motivo técnico baseado nos dados de entrada
    motivo_tecnico = "A análise da sua relação de renda, idade e valor da garantia/crédito solicitado não está alinhada com nossos critérios internos de política de crédito."

    system_prompt = """
    Você é um Analista de Relacionamento Sênior da Ventture. Sua tarefa é gerar uma mensagem de negativa de crédito com um tom extremamente empático, profissional e consultivo.
    O objetivo é manter o relacionamento com o cliente.

    A mensagem DEVE:
    1. Agradecer o interesse.
    2. Comunicar a decisão de não seguir adiante.
    3. Explicar a razão da negativa (baseada na entrada) de forma clara, mas suave.
    4. Oferecer-se para rever a situação no futuro.
    5. NÃO CITE NÚMEROS DE SCORE OU LIMITES. Fale de 'política de crédito' ou 'critérios internos'.
    6. Use a saudação "Prezado Cliente," no início.
    """

    user_prompt = f"""
    Motivo Técnico da Negativa (Use como base para a explicação): {motivo_tecnico}

    Gere a mensagem final completa e profissional.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3
            )
        )
        return response.text.strip()
    except Exception:
        return f"Erro: Não foi possível gerar a mensagem de negativa personalizada."


# === Endpoint principal ===
# === Endpoint principal ===
@app.post("/predict")
def prever_credito(data: CreditoRequest):
    try:
        # ... (criação do df_cliente, X_scaled, prob e aprovado)

        df_cliente = criar_features_cliente(
            renda=data.renda,
            idade=data.idade,
            valor_credito=data.valor_credito,
            valor_bem=data.valor_bem,
            liquidez=data.liquidez
        )

        X_scaled = scaler.transform(df_cliente)
        prob = modelo.predict_proba(X_scaled)[0][1]
        aprovado = int(prob >= 0.5)

        # === Geração da Mensagem LLM (NOVO) ===
        if aprovado:
            mensagem_cliente = gerar_mensagem_aprovacao(data, prob)
        else:
            mensagem_cliente = gerar_mensagem_negacao(data, prob)
        # ======================================

        # === Explicação ELI5 (código existente) ===
        perm = PermutationImportance(modelo, random_state=42)
        perm.fit(X_scaled, [int(prob >= 0.5)])
        html_obj = eli5.show_weights(perm, feature_names=df_cliente.columns.tolist(), top=8)
        explicacao_html = f"""
            <style>
                body {{ color: #1E90FF; font-family: Arial; }}
                table {{ color: #1E90FF; }}
            </style>
            {html_obj.data}
            """

        # Cria o registro
        novo_registro = {
            "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "renda": data.renda,
            "idade": data.idade,
            "valor_credito": data.valor_credito,
            "valor_bem": data.valor_bem,
            "liquidez": data.liquidez,
            "prob_aprovacao": round(float(prob), 4),
            "aprovado": "Sim" if aprovado else "Não",
            "explicacao_html": explicacao_html,
            "mensagem_cliente": mensagem_cliente  # NOVO CAMPO
        }

        # Gravar histórico (código existente)
        historico = pd.DataFrame([novo_registro])
        if os.path.exists(ARQUIVO_HISTORICO):
            historico_antigo = pd.read_csv(ARQUIVO_HISTORICO)
            historico = pd.concat([historico_antigo, historico], ignore_index=True)
        historico.to_csv(ARQUIVO_HISTORICO, index=False)

        return novo_registro

    except Exception as e:
        print(f"Erro interno: {e}")
        return {"error": "Erro interno no serviço", "detail": str(e)}


# === Endpoint para histórico ===
@app.get("/historico/")
def obter_historico():
    try:
        if not os.path.exists(ARQUIVO_HISTORICO):
            return []

        historico = pd.read_csv(ARQUIVO_HISTORICO)
        historico = historico.replace([np.inf, -np.inf], np.nan).fillna("")
        return historico.to_dict(orient="records")

    except Exception as e:
        print(f"Erro ao ler histórico: {e}")
        return {"error": "Falha ao obter histórico", "detail": str(e)}


# === Endpoint de teste ===
@app.get("/")
def root():
    return {"status": "Serviço de Crédito Ventture ativo!"}
