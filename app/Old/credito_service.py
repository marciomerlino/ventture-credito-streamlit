from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sys
import eli5
import os
import numpy as np
from eli5.sklearn import PermutationImportance

# === Configuração do FastAPI ===
app = FastAPI(title="Serviço de Crédito Ventture")

# === Definir caminhos ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_MODELO = os.path.join(BASE_DIR, "model", "modelo_credito.pkl")
ARQUIVO_SCALER = os.path.join(BASE_DIR, "model", "scaler.pkl")
HISTORICO_FILE = os.path.join(BASE_DIR, "historico_simulacoes.csv")

# === Carregar modelo e scaler ===
modelo = joblib.load(ARQUIVO_MODELO)
scaler = joblib.load(ARQUIVO_SCALER)

# === Adicionar src ao sys.path para import preprocess ===
sys.path.append(os.path.join(BASE_DIR, "src"))
from preprocess import criar_features_cliente

# === Modelos de dados ===
class Cliente(BaseModel):
    renda: float
    idade: int
    valor_credito: float
    valor_bem: float
    liquidez: str

# === Função para salvar no histórico ===
def salvar_historico(registro: pd.DataFrame):
    if os.path.exists(HISTORICO_FILE):
        historico = pd.read_csv(HISTORICO_FILE)
        historico = pd.concat([historico, registro], ignore_index=True)
    else:
        historico = registro
    historico.to_csv(HISTORICO_FILE, index=False)

# === Endpoint de previsão de crédito ===
@app.post("/prever_credito/")
def prever_credito(cliente: Cliente):
    df_cliente = criar_features_cliente(
        renda=cliente.renda,
        idade=cliente.idade,
        valor_credito=cliente.valor_credito,
        valor_bem=cliente.valor_bem,
        liquidez=cliente.liquidez
    )

    X_scaled = scaler.transform(df_cliente)
    prob = modelo.predict_proba(X_scaled)[0][1]
    aprovado = "Sim" if prob >= 0.5 else "Não"

    # === Explicação ELI5 ===
    perm = PermutationImportance(modelo, random_state=42)
    perm.fit(X_scaled, [int(prob>=0.5)])
    html_obj = eli5.show_weights(perm, feature_names=df_cliente.columns.tolist(), top=8)
    explicacao_html = f"""
    <style>
        body {{ color: #1E90FF; font-family: Arial; }}
        table {{ color: #1E90FF; }}
    </style>
    {html_obj.data}
    """

    # === Salvar no histórico ===
    novo_registro = pd.DataFrame([{
        "data_hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "renda": cliente.renda,
        "idade": cliente.idade,
        "valor_credito": cliente.valor_credito,
        "valor_bem": cliente.valor_bem,
        "liquidez": cliente.liquidez,
        "prob_aprovacao": round(prob,4),
        "aprovado": aprovado
    }])
    salvar_historico(novo_registro)

    return {
        "aprovado": aprovado,
        "probabilidade_aprovacao": prob,
        "explicacao_html": explicacao_html
    }

# === Endpoint para consultar histórico ===
#@app.get("/historico/")
#def consultar_historico():
#    if os.path.exists(HISTORICO_FILE):
#        historico = pd.read_csv(HISTORICO_FILE)
#        return historico.to_dict(orient="records")
#    else:
#        return []

@app.get("/historico/")
def historico():
    try:
        if os.path.exists(HISTORICO_FILE):
            df = pd.read_csv(HISTORICO_FILE)
            # 🔧 Substitui NaN por None para evitar erro de JSON
            df = df.replace({np.nan: None})
            return df.to_dict(orient="records")
        else:
            return []
    except Exception as e:
        return {"error": str(e)}


# === Endpoint para limpar histórico ===
@app.post("/limpar_historico/")
def limpar_historico():
    if os.path.exists(HISTORICO_FILE):
        os.remove(HISTORICO_FILE)
    return {"status":"Histórico apagado"}
