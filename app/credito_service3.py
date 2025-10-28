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


# === Endpoint principal ===
@app.post("/predict")
def prever_credito(data: CreditoRequest):
    try:
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

        # === Explicação ELI5 ===
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
            "explicacao_html": explicacao_html
        }

        # Gravar histórico
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
