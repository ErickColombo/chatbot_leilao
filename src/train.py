import json
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "faq.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "modelo.pkl"


# ============================================================
# CARREGAR DATASET
# ============================================================

print("=" * 60)
print("TREINAMENTO DO CHATBOT")
print("=" * 60)

print("\nCarregando dataset...")

if not DATA_PATH.exists():
    print(f"\nERRO: Dataset não encontrado:")
    print(DATA_PATH)
    raise SystemExit(1)


with open(DATA_PATH, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)


# ============================================================
# VALIDAR JSON
# ============================================================

if "intents" not in dados:
    print("\nERRO: O arquivo faq.json precisa possuir 'intents'.")
    raise SystemExit(1)


intents = dados["intents"]

if not intents:
    print("\nERRO: Nenhuma intent encontrada.")
    raise SystemExit(1)


# ============================================================
# PREPARAR DADOS
# ============================================================

perguntas = []
labels = []
respostas = {}

print("\nPreparando perguntas...")

for item in intents:

    if "intent" not in item:
        print("AVISO: Uma intent não possui o campo 'intent'.")
        continue

    if "perguntas" not in item:
        print(
            f"AVISO: Intent '{item['intent']}' "
            "não possui perguntas."
        )
        continue

    if "resposta" not in item:
        print(
            f"AVISO: Intent '{item['intent']}' "
            "não possui resposta."
        )
        continue

    intent = item["intent"]

    resposta = item["resposta"]

    respostas[intent] = resposta

    for pergunta in item["perguntas"]:

        if not isinstance(pergunta, str):
            continue

        pergunta = pergunta.strip()

        if not pergunta:
            continue

        perguntas.append(pergunta)
        labels.append(intent)


# ============================================================
# VALIDAÇÃO
# ============================================================

print(f"\nTotal de perguntas: {len(perguntas)}")
print(f"Total de intents: {len(respostas)}")

if len(perguntas) == 0:
    print("\nERRO: Nenhuma pergunta foi encontrada.")
    raise SystemExit(1)


if len(respostas) < 2:
    print(
        "\nERRO: O SVM precisa de pelo menos "
        "duas intents diferentes."
    )
    raise SystemExit(1)


# ============================================================
# MOSTRAR INTENTS
# ============================================================

print("\nIntents encontradas:")

for intent, quantidade in sorted(
    {
        intent: labels.count(intent)
        for intent in set(labels)
    }.items()
):

    print(
        f"  - {intent}: {quantidade} perguntas"
    )


# ============================================================
# CRIAR MODELO
# ============================================================

print("\nCriando modelo TF-IDF + SVM...")


modelo = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",

                # Palavras individuais + pares de palavras
                ngram_range=(1, 2),

                # Ignora termos extremamente raros
                min_df=1,

                # Limite razoável para o vocabulário
                sublinear_tf=True
            )
        ),

        (
            "svm",
            LinearSVC(
                C=1.5,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# TREINAMENTO
# ============================================================

print("\nTreinando modelo...")

modelo.fit(
    perguntas,
    labels
)


# ============================================================
# CRIAR DIRETÓRIO DOS MODELOS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPARAR OBJETO DO MODELO
# ============================================================

dados_modelo = {
    "modelo": modelo,

    "respostas": respostas,

    "desambiguacoes": dados.get(
        "desambiguacoes",
        []
    )
}


# ============================================================
# SALVAR
# ============================================================

joblib.dump(
    dados_modelo,
    MODEL_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TREINAMENTO CONCLUÍDO")
print("=" * 60)

print(f"\nModelo salvo em:")

print(MODEL_PATH)

print("\nO chatbot está pronto para ser executado.")
