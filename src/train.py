import json
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ==========================================
# CAMINHOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "faq.json"
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"


# ==========================================
# CARREGAR DATASET
# ==========================================

print("Carregando dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)


# ==========================================
# PREPARAR DADOS
# ==========================================

perguntas = []
intents = []
respostas = {}


for item in dados:

    intent = item["intent"]
    resposta = item["resposta"]

    # Guarda a resposta associada à intenção
    respostas[intent] = resposta

    # Adiciona todas as perguntas daquela intenção
    for pergunta in item["perguntas"]:
        perguntas.append(pergunta)
        intents.append(intent)


print(f"Total de perguntas: {len(perguntas)}")
print(f"Total de intents: {len(set(intents))}")


# ==========================================
# CRIAR MODELO
# ==========================================

print("\nCriando modelo TF-IDF + SVM...")

modelo = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "svm",
        LinearSVC()
    )
])


# ==========================================
# TREINAR
# ==========================================

print("Treinando modelo...")

modelo.fit(perguntas, intents)


# ==========================================
# SALVAR MODELO
# ==========================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "modelo": modelo,
        "respostas": respostas
    },
    MODEL_PATH
)


print("\nModelo treinado com sucesso!")

print(f"Modelo salvo em:")
print(MODEL_PATH)
