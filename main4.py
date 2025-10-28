from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
from dotenv import load_dotenv

# Importações do SDK puro do Google Gemini
from google import genai
from google.genai.errors import APIError


# --- Funções de Carregamento de Dados ---

def carregar_json(nome_arquivo: str):
    """Carrega dados de um arquivo JSON."""
    try:
        with open(nome_arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo {nome_arquivo} não encontrado.")
        return []
    except json.JSONDecodeError as e:
        print(f"ERRO: Formato JSON inválido no arquivo {nome_arquivo}: {e}")
        return []
    except Exception as e:
        print(f"ERRO: Falha ao carregar {nome_arquivo}: {e}")
        return []


# --- 1. Modelos de Dados Pydantic ---

class ProdutoCredito(BaseModel):
    """Detalhes de um produto de crédito que o banco pode oferecer (para catálogo)."""
    id_produto: str
    nome: str
    taxa_base_anual: float
    prazo_max_meses: int
    limite_max_inicial: float
    requisito_score_min: int
    requisito_garantia: str


class ClienteBase(BaseModel):
    """Informações detalhadas do cliente (para base de dados simulada)."""
    id_cliente: str
    idade: int
    score_interno_risco: int
    tempo_relacionamento_anos: int
    saldo_total_investimentos: float
    possui_imovel_rural: bool
    historico_inadimplencia: bool
    necessidade_financiamento: float = 0.0  # Campo que será populado dinamicamente


class DecisaoInput(BaseModel):
    """Modelo de entrada para o endpoint de Decisão Simples (não usado no fluxo único, mas mantido)."""
    id_cliente: str
    necessidade_financiamento: float


class FluxoUnicoInput(BaseModel):
    """Dados de entrada para o endpoint de fluxo único (Decisão + GenAI)."""
    id_cliente: str
    necessidade_financiamento: float
    finalidade_credito: str


class OfertaGerada(BaseModel):
    """A melhor oferta final gerada pelo motor de Decisão (usada internamente)."""
    status: str
    mensagem: str
    produto_ofertado: str
    taxa_final_anual: float
    limite_aprovado: float
    prazo_meses: int
    motivo_recomendacao: str


class DadosCreditoGenAI(BaseModel):
    """Modelo de entrada para a função de Geração de Texto (GenAI)."""
    id_cliente: str
    limite_aprovado: float
    taxa_juros_anual: float
    prazo_meses: int
    relacionamento_chave: str
    finalidade_credito: str
    garantias_oferecidas: str


# --- 2. Carregamento de Chave e Bases de Conhecimento ---

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Carregar a Base de Conhecimento (Simulação RAG)
try:
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_CONTEXT = f.read()
    print("Base de conhecimento (RAG) carregada.")
except FileNotFoundError:
    KNOWLEDGE_CONTEXT = "Nenhuma base de conhecimento encontrada."
    print("AVISO: knowledge_base.txt não encontrado.")

# Carregar bases de dados simuladas (JSON)
PRODUTOS_SIMULADOS_DATA = carregar_json("produtos_simulados.json")
CLIENTES_SIMULADOS_DATA = carregar_json("clientes_simulados.json")

PRODUTOS_OFERECIDOS = [ProdutoCredito(**p) for p in PRODUTOS_SIMULADOS_DATA]
CLIENTES_DB = {c['id_cliente']: ClienteBase(**c) for c in CLIENTES_SIMULADOS_DATA}

print(f"Carregados {len(PRODUTOS_OFERECIDOS)} produtos e {len(CLIENTES_DB)} clientes.")

# Inicialização do Cliente Gemini
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        print("Cliente Gemini inicializado.")
    except Exception as e:
        print(f"ERRO CRÍTICO na inicialização do Cliente Gemini: {e}")
        client = None
else:
    print("ERRO: GEMINI_API_KEY não encontrada. A funcionalidade GenAI estará desativada.")


# --- 3. Motor de Decisão (Simulação de ML/Heurística) ---

def motor_de_decisao(cliente: ClienteBase, produtos_catalogo: list[ProdutoCredito]) -> OfertaGerada:
    """
    Simula um modelo de decisão que define o melhor produto, limite e taxa.
    """
    produtos_elegiveis = []

    # 1. Pré-seleção: Verifica elegibilidade
    for produto in produtos_catalogo:
        if cliente.score_interno_risco >= produto.requisito_score_min:
            if produto.requisito_garantia == "Nenhuma" or \
                    (produto.requisito_garantia == "Imóvel Rural" and cliente.possui_imovel_rural):
                produtos_elegiveis.append(produto)

    if not produtos_elegiveis:
        return OfertaGerada(
            status="NEGADO",
            mensagem="Cliente não atende aos requisitos de Score e/ou Garantia para nossos produtos.",
            produto_ofertado="", taxa_final_anual=0.0, limite_aprovado=0.0, prazo_meses=0,
            motivo_recomendacao="Baixo Score ou falta de garantias exigidas."
        )

    # 2. Seleção do Melhor Produto (Heurística: Menor Taxa Base)
    melhor_produto = min(produtos_elegiveis, key=lambda p: p.taxa_base_anual)

    # 3. Cálculo da Oferta e Bônus

    # Limite Aprovado:
    ajuste_risco = cliente.score_interno_risco / 1000.0
    limite_ajustado = melhor_produto.limite_max_inicial * ajuste_risco

    limite_aprovado = min(
        cliente.necessidade_financiamento * 1.05,
        melhor_produto.limite_max_inicial,
        limite_ajustado
    )
    limite_aprovado = max(0, limite_aprovado)

    # Taxa Final:
    taxa_final = melhor_produto.taxa_base_anual
    motivos = [f"Produto base: {melhor_produto.nome} ({taxa_final:.2f}% a.a.)"]

    if cliente.tempo_relacionamento_anos >= 10:
        taxa_final -= 0.5
        motivos.append("Bônus Fidelidade (-0.5 p.p.)")

    if cliente.saldo_total_investimentos >= 200000.00:
        taxa_final -= 0.25
        motivos.append("Bônus Investimento (-0.25 p.p.)")

    # Prazo (Ajuste simplificado)
    prazo_final = min(int(cliente.necessidade_financiamento / 10000) * 12, melhor_produto.prazo_max_meses)
    prazo_final = max(24, prazo_final)

    return OfertaGerada(
        status="APROVADO",
        mensagem="Oferta otimizada gerada com sucesso.",
        produto_ofertado=melhor_produto.nome,
        taxa_final_anual=round(taxa_final, 2),
        limite_aprovado=round(limite_aprovado, 2),
        prazo_meses=prazo_final,
        motivo_recomendacao=" | ".join(motivos)
    )


# --- 4. Lógica de Geração de Texto (GenAI com RAG) ---

def gerar_proposta_valor_com_contexto(dados: DadosCreditoGenAI) -> str:
    """
    Função que usa o contexto de conhecimento lido do arquivo para guiar o Gemini.
    """
    if not client:
        # Esta exceção só deve ser lançada se o cliente for chamado e não estiver inicializado.
        raise Exception("Cliente Gemini não inicializado.")

    system_prompt = """
    Você é um Gerente de Crédito Agrário sênior do Banco Alpha. Sua tarefa é criar uma 
    proposta de valor consultiva, clara e que utilize as regras internas.
    O tom deve ser profissional e focado em valor para o cliente.
    """

    contexto_prompt = f"""
    CONTEXTO DE CONHECIMENTO DO BANCO (OBRIGATÓRIO USAR PARA REFERÊNCIA):
    ---
    {KNOWLEDGE_CONTEXT}
    ---
    """

    dados_cliente_prompt = f"""
    INFORMAÇÕES DO CLIENTE PARA A PROPOSTA:
    - Cliente ID: {dados.id_cliente}
    - Relacionamento: {dados.relacionamento_chave}
    - Limite Aprovado: R$ {dados.limite_aprovado:,.2f}
    - Taxa: {dados.taxa_juros_anual:.2f}% a.a.
    - Prazo: {dados.prazo_meses} meses
    - Finalidade: {dados.finalidade_credito}
    - Garantias: {dados.garantias_oferecidas}

    Com base nas informações acima e no CONTEXTO DE CONHECIMENTO, gere a proposta final de valor.
    """

    full_prompt = contexto_prompt + dados_cliente_prompt

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4
            )
        )

        return response.text.strip()

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Erro na API Gemini: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado durante a geração: {str(e)}")


def gerar_negativa_empática(id_cliente: str, motivo_tecnico: str) -> str:
    """
    Função que usa o Gemini para criar uma resposta empática em caso de negativa.
    """
    if not client:
        raise Exception("Cliente Gemini não inicializado.")

    system_prompt = """
    Você é um Analista de Relacionamento Sênior do Banco Alpha, responsável por comunicar decisões de crédito. 
    Sua tarefa é gerar uma mensagem de negativa de crédito com um tom extremamente empático, profissional e consultivo. 
    O objetivo é manter o relacionamento com o cliente.

    A mensagem DEVE:
    1. Agradecer o interesse.
    2. Comunicar a decisão de não seguir adiante.
    3. Explicar a razão da negativa (baseada na entrada) de forma clara, mas suave.
    4. Oferecer-se para rever a situação no futuro.
    5. NÃO CITE NÚMEROS DE SCORE OU LIMITES. Fale de 'política de crédito' ou 'critérios internos'.
    """

    user_prompt = f"""
    Cliente ID: {id_cliente}
    Motivo Técnico da Negativa: {motivo_tecnico}

    Gere a mensagem final completa e profissional.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3  # Temperatura mais baixa para garantir um tom seguro e consistente
            )
        )

        return response.text.strip()

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Erro na API Gemini: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado durante a geração da negativa: {str(e)}")

# --- 5. Configuração e Endpoints do FastAPI ---

app = FastAPI(title="GenAI Crédito Agrário Prototype V6.0 (Fluxo Único)")


# --- NOVO ENDPOINT DE FLUXO ÚNICO ---
@app.post("/proposta_completa/gerar")
def gerar_proposta_completa_api(input_data: FluxoUnicoInput):
    """
    Endpoint de fluxo único:
    1. Executa o Motor de Decisão (ML).
    2. Usa o resultado para gerar o texto final via GenAI/RAG.
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="O Cliente Gemini não foi inicializado. Verifique a variável GEMINI_API_KEY no .env."
        )

    # --------------------------------------------------------
    # PASSO 1: DECISÃO (Motor de Decisão/ML)
    # --------------------------------------------------------
    if input_data.id_cliente not in CLIENTES_DB:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {input_data.id_cliente} não encontrado na base de dados simulada."
        )

    cliente_base = CLIENTES_DB[input_data.id_cliente].model_copy(
        update={'necessidade_financiamento': input_data.necessidade_financiamento})

    try:
        oferta_decisao = motor_de_decisao(cliente_base, PRODUTOS_OFERECIDOS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Motor de Decisão: {str(e)}")

    if oferta_decisao.status == "NEGADO":
        # CHAMA O GEMINI PARA GERAR A RESPOSTA EMPÁTICA
        try:
            texto_negativa = gerar_negativa_empática(
                id_cliente=input_data.id_cliente,
                motivo_tecnico=oferta_decisao.mensagem  # A mensagem técnica do motor de decisão
            )
        except Exception as e:
            # Em caso de falha no GenAI, retorna a mensagem técnica de forma segura
            texto_negativa = f"Prezado(a) cliente, lamentamos informar que não foi possível prosseguir com sua solicitação neste momento. Motivo: {oferta_decisao.mensagem}"
            print(f"AVISO: Falha no GenAI de negativa, retornando mensagem fallback. Erro: {e}")

        return {
            "status": "NEGADO",
            "id_transacao": input_data.id_cliente + "_FLUXO_NEGATIVA",
            "proposta_valor_texto": texto_negativa  # Retorna o texto gerado pelo GenAI
        }

    # --------------------------------------------------------
    # PASSO 2: GERAÇÃO (GenAI/RAG)
    # --------------------------------------------------------

    # Mapear a saída da Decisão (oferta_decisao) para a entrada do GenAI
    garantia = "Imóvel Rural" if cliente_base.possui_imovel_rural else "Garantias diversas/Fiança"

    dados_genai = DadosCreditoGenAI(
        id_cliente=input_data.id_cliente,
        limite_aprovado=oferta_decisao.limite_aprovado,
        taxa_juros_anual=oferta_decisao.taxa_final_anual,
        prazo_meses=oferta_decisao.prazo_meses,
        relacionamento_chave=f"Produto Recomendado: {oferta_decisao.produto_ofertado}",
        finalidade_credito=input_data.finalidade_credito,
        garantias_oferecidas=garantia
    )

    try:
        proposta_texto = gerar_proposta_valor_com_contexto(dados_genai)

        return {
            "status": "APROVADO",
            "id_transacao": input_data.id_cliente + "_FLUXO_UNICO_" + str(len(proposta_texto)),
            "produto_ofertado": oferta_decisao.produto_ofertado,
            "proposta_valor_texto": proposta_texto
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha na Geração GenAI: {str(e)}")


# --- ENDPOINTS ANTIGOS (Mantidos para debug/teste isolado) ---

@app.post("/decisao_ml/melhor_oferta", response_model=OfertaGerada)
def decidir_melhor_oferta_api(input_data: DecisaoInput):
    """
    [DEBUG/LEGADO] Simula o Motor de Decisão para teste isolado.
    """
    if input_data.id_cliente not in CLIENTES_DB:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {input_data.id_cliente} não encontrado na base de dados simulada."
        )
    cliente_base = CLIENTES_DB[input_data.id_cliente].model_copy(
        update={'necessidade_financiamento': input_data.necessidade_financiamento})
    try:
        oferta = motor_de_decisao(cliente_base, PRODUTOS_OFERECIDOS)
        return oferta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no motor de decisão: {str(e)}")


@app.post("/proposta_genai/gerar")
def gerar_proposta_api(dados: DadosCreditoGenAI):
    """
    [DEBUG/LEGADO] Gera texto GenAI a partir de condições pré-definidas (para teste isolado).
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="O Cliente Gemini não foi inicializado. Verifique a variável GEMINI_API_KEY no .env."
        )
    try:
        proposta_texto = gerar_proposta_valor_com_contexto(dados)
        return {
            "status": "SUCESSO",
            "id_transacao": dados.id_cliente + "_SDK_RAG_" + str(len(proposta_texto)),
            "proposta_valor_texto": proposta_texto
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha interna ao gerar a proposta: {str(e)}")


# --- 6. Execução do Servidor ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)