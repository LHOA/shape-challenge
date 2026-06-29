# Shape AI/ML Challenge — Predictive Maintenance para FPSO

Este repositório contém a solução de **Luis Henrique Oliveira Assunção** para o desafio de Machine Learning da Shape Digital. O projeto é focado na manutenção preditiva de equipamentos em um FPSO (*Floating Production, Storage, and Offloading*).

## 📌 O Desafio
O objetivo principal é analisar séries temporais de sensores (temperatura, pressão, vibração, frequência) e configurações operacionais (Presets) para:
1. Identificar eventos reais de falha.
2. Descobrir a causa-raiz (sensores que indicam anomalias precoces).
3. Desenvolver um modelo preditivo (XGBoost) capaz de alertar sobre falhas antes que elas ocorram (sem *data leakage*).
4. Analisar a importância das variáveis utilizando SHAP.

## 🗂 Estrutura do Projeto
O projeto foi desenvolvido focando em **boas práticas de Engenharia de Software e Machine Learning**, separando a exploração visual da lógica de negócio.

* `notebooks/`: Contém o notebook principal (`shape_challenge.ipynb`) com o relatório técnico, *storytelling* dos dados e a discussão sobre arquitetura para produção.
* `src/`: Módulos Python em nível de produção (com *type-hints* rigorosos).
  * `eda.py`: Funções para validação de dados e extração de eventos de falha.
  * `features.py`: Feature engineering (variáveis de *lag*, janelas móveis e interações).
  * `models.py`: Treinamento (XGBoost tunado e Logistic Regression baseline) e validação cruzada temporal (*TimeSeriesSplit*).
  * `visualization.py`: Funções padronizadas para geração de gráficos.
* `output/`: Gráficos exportados e o relatório HTML final gerado.

## 🛠 Tecnologias e Qualidade de Código
- **Stack de ML:** Python 3.11, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn.
- **Code Quality:** Todo o código na pasta `src/` foi formatado e validado com:
  - **`black`** (Code formatter)
  - **`ruff`** (Linter ultrarrápido)
  - **`mypy`** (Checagem estática de tipagem estrita)

## 🚀 Como Executar Localmente
1. Clone este repositório:
   ```bash
   git clone https://github.com/LHOA/shape-challenge.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloque a base de dados original em `data/raw/Test O_G_Equipment_Data.xlsx`. *(Nota: o diretório de dados está ignorado no `.gitignore` por boas práticas e governança de dados).*
4. Execute o notebook `notebooks/shape_challenge.ipynb` para visualizar a análise completa.
