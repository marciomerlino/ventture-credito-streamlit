import pandas as pd

def criar_features_cliente(renda_mensal, idade, valor_solicitado, valor_bem, liquidez):
    liquidez_score = {"baixa": 1, "media": 2, "alta": 3}[liquidez]
    relacao_garantia_credito = valor_bem / (valor_solicitado + 1)
    renda_por_idade = renda_mensal / (idade + 1)
    garantia_ponderada = relacao_garantia_credito * liquidez_score

    df = pd.DataFrame([{
        "renda_mensal": renda_mensal,
        "idade": idade,
        "valor_solicitado": valor_solicitado,
        "valor_bem": valor_bem,
        "relacao_garantia_credito": relacao_garantia_credito,
        "liquidez_score": liquidez_score,
        "renda_por_idade": renda_por_idade,
        "garantia_ponderada": garantia_ponderada
    }])
    return df
