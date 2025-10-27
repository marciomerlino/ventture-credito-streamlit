import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Carregar CSVs de teste
clientes = pd.read_csv("data/clientes.csv")
creditos = pd.read_csv("data/creditos.csv")
garantias = pd.read_csv("data/garantias.csv")

# Merge simples (exemplo)
df = clientes.merge(creditos, left_on="id_cliente", right_on="id_cliente") \
             .merge(garantias, left_on="id_cliente", right_on="id_cliente")

# Criar features básicas
df['relacao_garantia_credito'] = df['valor_bem'] / (df['valor_solicitado'] + 1)
df['liquidez_score'] = df['liquidez'].map({'baixa': 1, 'media':2, 'alta':3})
df['renda_por_idade'] = df['renda_mensal'] / (df['idade'] + 1)
df['garantia_ponderada'] = df['relacao_garantia_credito'] * df['liquidez_score']

features = ['renda_mensal','idade','valor_solicitado','valor_bem','relacao_garantia_credito',
            'liquidez_score','renda_por_idade','garantia_ponderada']

X = df[features]
y = df['aprovado'].map({'Sim':1,'Não':0})

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_scaled, y)

# Salvar
joblib.dump(modelo, "model/modelo_credito.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Modelo e scaler salvos em 'model/'")
