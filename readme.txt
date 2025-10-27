📘 Ventture Consulting – Aplicação de Análise de Crédito

💡 Descrição

Aplicação desenvolvida em Python + Streamlit para simulação e explicação de decisões de crédito.
Utiliza modelos de Machine Learning (Scikit-learn), explicabilidade com ELI5, e visual personalizado com o tema Ventture.

🧱 Estrutura do Projeto
📦 ModeloCredito
├── app/
│   └── app_credito.py          # Código principal da aplicação
├── data/
│   ├── clientes.csv            # Dados de clientes
│   ├── creditos.csv            # Dados de crédito
│   └── garantias.csv           # Dados de garantias
├── model/
│   ├── modelo_credito.pkl      # Modelo de predição treinado
│   └── scaler.pkl              # Escalonador de variáveis
├── assets/
│   └── ventture.jpg            # Logotipo Ventture
├── streamlit/
│   └── config.toml             # Tema personalizado Ventture
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo

⚙️ Execução Local

Crie um ambiente virtual (opcional, mas recomendado):

python -m venv .venv
.venv\Scripts\activate


Instale as dependências:

pip install -r requirements.txt


Execute o aplicativo:

streamlit run app/app_credito.py


O Streamlit abrirá automaticamente em
👉 http://localhost:8501

☁️ Deploy no Streamlit Cloud

Faça login em streamlit.io/cloud

Clique em “New app”

Selecione o repositório do projeto

Configure:

Main file path: app/app_credito.py

Branch: main

Clique em Deploy

🟢 O Streamlit Cloud detectará automaticamente o requirements.txt na raiz e aplicará o tema definido em streamlit/config.toml.

🎨 Tema Ventture (config.toml)
[theme]
primaryColor = "#004AAD"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#000000"
font = "sans serif"

🧠 Principais Tecnologias

Streamlit – Interface web interativa

Scikit-learn – Treinamento e predição de modelos de crédito

ELI5 – Explicação das decisões do modelo

Plotly / Matplotlib / Seaborn – Visualização dos resultados

Joblib – Persistência dos modelos treinados

👨‍💼 Desenvolvido por

Ventture Consulting

Consultoria em tecnologia, eficiência e automação para negócios inteligentes.