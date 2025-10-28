# app/credito_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import traceback
from typing import Any, Dict
from types import SimpleNamespace

# Tentativa de importar ELI5 (opcional; fallback se não existir)
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except Exception:
    ELI5_AVAILABLE = False

# ======================================================
# Ajuste BASE_DIR para apontar para a raiz do projeto
# (este arquivo está em app/, então subimos 1 nível)
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ARQUIVO_MODELO = os.path.join(MODEL_DIR, "modelo_credito.pkl")
ARQUIVO_SCALER = os.path.join(MODEL_DIR, "scaler.pkl")
HISTORICO_FILE = os.path.join(BASE_DIR, "historico_simulacoes.csv")

# ======================================================
# Carrega modelo e scaler (gera erro óbvio se não existir)
# ======================================================
modelo = joblib.load(ARQUIVO_MODELO)
scaler = joblib.load(ARQUIVO_SCALER)

# ======================================================
# Importa as funções de geração de mensagem (main4.py)
# ======================================================
# Espera-se que main4.py esteja na raiz do projeto. Se estiver em outra pasta,
# ajuste o sys.path ou mova o arquivo.
try:
    sys.path.append(BASE_DIR)
    # tenta importar as funções com e sem acento no nome (por segurança)
    from main4 import gerar_proposta_valor_com_contexto
    try:
        # função de negativa pode ter ou não acento no nome; tentamos ambas
        try:
            from main4 import gerar_negativa_empática as gerar_negativa_empatica
        except Exception:
            from main4 import gerar_negativa_empatica
    except Exception:
        # se não existir, definimos placeholder que será ignorado mais abaixo
        gerar_negativa_empatica = None
except Exception:
    # se import falhar, deixamos as funções como None e trataremos mais adiante
    gerar_proposta_valor_com_contexto = None
    gerar_negativa_empatica = None

# ======================================================
# FastAPI app
# ======================================================
app = FastAPI(title="Serviço de Crédito Ventture (Gemini + Modelo)")

# ======================================================
# Pydantic request model — usamos nomes simples esperados pelo app
# ======================================================
class Cliente(BaseModel):
    renda: float
    idade: int
    valor_credito: float
    valor_bem: float
    liquidez: str  # "baixa", "media", "alta"

# ======================================================
# Util: criar features (mantém compatibilidade com preprocess.py)
# ======================================================
def criar_features_cliente(renda, idade, valor_solicitado, valor_bem, liquidez):
    liquidez_score = {"baixa": 1, "media": 2, "alta": 3}.get(liquidez, 2)
    relacao_garantia_credito = float(valor_bem) / (float(valor_solicitado) + 1.0)
    renda_por_idade = float(renda) / (int(idade) + 1.0)
    garantia_ponderada = relacao_garantia_credito * liquidez_score
    df = pd.DataFrame([{
        "renda_mensal": float(renda),
        "idade": int(idade),
        "valor_solicitado": float(valor_solicitado),
        "valor_bem": float(valor_bem),
        "relacao_garantia_credito": relacao_garantia_credito,
        "liquidez_score": liquidez_score,
        "renda_por_idade": renda_por_idade,
        "garantia_ponderada": garantia_ponderada
    }])
    return df

# ======================================================
# Salvar histórico (append)
# ======================================================
def salvar_historico(registro: pd.DataFrame):
    if os.path.exists(HISTORICO_FILE):
        try:
            historico = pd.read_csv(HISTORICO_FILE)
            historico = pd.concat([historico, registro], ignore_index=True)
        except Exception:
            # se CSV corrompido, reescreve com novo registro
            historico = registro
    else:
        historico = registro
    historico.to_csv(HISTORICO_FILE, index=False)

# ======================================================
# Endpoint: prever_credito
# - gera probabilidade, aprovado/negado
# - chama Gemini via suas funções em main4.py para criar mensagem
# - gera explicacao HTML com ELI5 (opcional/fallback)
# - salva histórico incluindo mensagem
# ======================================================
@app.post("/prever_credito/")
def prever_credito(cliente: Cliente) -> Dict[str, Any]:
    try:
        # 1) features
        df_cliente = criar_features_cliente(
            renda=cliente.renda,
            idade=cliente.idade,
            valor_solicitado=cliente.valor_credito,
            valor_bem=cliente.valor_bem,
            liquidez=cliente.liquidez
        )

        # 2) normalizar e prever
        X_scaled = scaler.transform(df_cliente)
        prob = float(modelo.predict_proba(X_scaled)[0][1])
        aprovado = True if prob >= 0.5 else False
        aprovado_str = "Sim" if aprovado else "Não"

        # 3) gerar mensagem via Gemini (usando as funções de main4.py)
        mensagem_cliente = None
        try:
            if aprovado and callable(gerar_proposta_valor_com_contexto):
                # passamos contexto básico: cliente + probabilidade
                contexto = {
                    "renda": cliente.renda,
                    "idade": cliente.idade,
                    "valor_credito": cliente.valor_credito,
                    "valor_bem": cliente.valor_bem,
                    "liquidez": cliente.liquidez,
                    "probabilidade": prob
                }
                # sua função deve retornar string pronta para enviar ao cliente
                mensagem_cliente = gerar_proposta_valor_com_contexto(SimpleNamespace(**contexto))
            elif (not aprovado) and callable(gerar_negativa_empatica):
                contexto = {
                    "renda": cliente.renda,
                    "idade": cliente.idade,
                    "valor_credito": cliente.valor_credito,
                    "valor_bem": cliente.valor_bem,
                    "liquidez": cliente.liquidez,
                    "probabilidade": prob
                }
                mensagem_cliente = gerar_negativa_empatica(SimpleNamespace(**contexto))
        except Exception as e:
            # se houver erro ao chamar a função Gemini, registramos e caímos no fallback abaixo
            mensagem_cliente = None
            app.logger = getattr(app, "logger", None)
            # optional: log traceback
            print("Erro ao gerar mensagem Gemini:", str(e))
            traceback.print_exc()

        # 4) explicação com ELI5 (opcional). ELI5 precisa de mais de uma amostra; usamos fallback simples se não possível.
        explicacao_html = None
        if ELI5_AVAILABLE:
            try:
                # PermutationImportance precisa de Y; aqui usamos prob threshold como proxy para label local.
                perm = PermutationImportance(modelo, random_state=42)
                # NOTE: perm.fit expects X (n_samples x n_features) and y (n_samples,)
                # If we have only one sample, permutation importance won't be meaningful.
                # We'll try to use historico (if available) to compute perm importance; fallback para X_scaled if necessary.
                if os.path.exists(HISTORICO_FILE):
                    try:
                        df_hist = pd.read_csv(HISTORICO_FILE)
                        # Build X_hist features to fit perm: attempt to map columns to training names
                        # If columns match, use them; else fallback to scaled single sample
                        feature_cols = df_cliente.columns.tolist()
                        if set(feature_cols).issubset(set(df_hist.columns)):
                            X_hist = df_hist[feature_cols].values
                            y_hist = df_hist['aprovado'].map({'Sim':1,'Não':0}).values
                            perm.fit(X_hist, y_hist)
                        else:
                            # fallback: try fit on the single scaled sample (will produce limited result)
                            perm.fit(X_scaled, [int(aprovado)])
                    except Exception:
                        perm.fit(X_scaled, [int(aprovado)])
                else:
                    perm.fit(X_scaled, [int(aprovado)])
                html_obj = eli5.show_weights(perm, feature_names=df_cliente.columns.tolist(), top=8)
                explicacao_html = html_obj.data
            except Exception:
                explicacao_html = None

        # 5) fallback message if Gemini function not available or failed
        if not mensagem_cliente:
            if aprovado:
                mensagem_cliente = (
                    f"Parabéns — seu crédito foi aprovado com probabilidade estimada de {prob:.2%}. "
                    "Em breve entraremos em contato para formalizar a proposta."
                )
            else:
                mensagem_cliente = (
                    f"Infelizmente seu pedido não foi aprovado (probabilidade estimada {prob:.2%}). "
                    "Podemos conversar sobre alternativas e próximas etapas."
                )

        # 6) salvar no histórico incluindo a mensagem gerada
        novo_registro = pd.DataFrame([{
            "data_hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "renda": cliente.renda,
            "idade": cliente.idade,
            "valor_credito": cliente.valor_credito,
            "valor_bem": cliente.valor_bem,
            "liquidez": cliente.liquidez,
            "prob_aprovacao": round(prob, 4),
            "aprovado": aprovado_str,
            "mensagem_cliente": mensagem_cliente
        }])
        salvar_historico(novo_registro)

        # 7) resposta
        response = {
            "aprovado": aprovado_str,
            "probabilidade_aprovacao": prob,
            "mensagem_cliente": mensagem_cliente
        }
        if explicacao_html:
            response["explicacao_html"] = explicacao_html

        return response

    except Exception as e:
        # erro geral
        traceback.print_exc()
        return {"error": "Erro interno no serviço", "detail": str(e)}

# ======================================================
# Endpoint: obter histórico
# ======================================================
@app.get("/historico/")
def consultar_historico():
    if os.path.exists(HISTORICO_FILE):
        try:
            historico = pd.read_csv(HISTORICO_FILE)
            # Retornar como lista de registros JSON
            return historico.to_dict(orient="records")
        except Exception:
            return {"error": "Falha ao ler histórico"}
    else:
        return []

# ======================================================
# Endpoint: limpar histórico
# ======================================================
@app.post("/limpar_historico/")
def limpar_historico():
    try:
        if os.path.exists(HISTORICO_FILE):
            os.remove(HISTORICO_FILE)
        return {"status": "Histórico apagado"}
    except Exception as e:
        return {"error": "Falha ao apagar histórico", "detail": str(e)}
