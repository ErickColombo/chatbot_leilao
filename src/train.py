import json
import os
import re
import unicodedata
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "faq.json"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "modelo.pkl"
)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(texto):
    """
    Normaliza o texto para facilitar comparação de perguntas.

    Exemplo:
        "Onde mudo minha senha?"
        ->
        "onde mudo minha senha"
    """

    if not isinstance(texto, str):
        return ""

    texto = texto.lower().strip()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = re.sub(
        r"[^\w\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# ============================================================
# CARREGAR JSON
# ============================================================

def carregar_dataset():

    print("=" * 60)
    print("CARREGANDO DATASET")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{DATASET_PATH}"
        )

    try:
        with open(
            DATASET_PATH,
            "r",
            encoding="utf-8"
        ) as arquivo:

            dados = json.load(arquivo)

    except json.JSONDecodeError as erro:

        raise ValueError(
            f"\nERRO DE JSON!\n"
            f"Linha: {erro.lineno}\n"
            f"Coluna: {erro.colno}\n"
            f"Mensagem: {erro.msg}\n"
        )

    if not isinstance(dados, dict):
        raise ValueError(
            "O JSON principal precisa ser um objeto/dict."
        )

    if "intents" not in dados:
        raise ValueError(
            "O JSON não possui a chave 'intents'."
        )

    if "desambiguacoes" not in dados:
        raise ValueError(
            "O JSON não possui a chave 'desambiguacoes'."
        )

    if not isinstance(dados["intents"], list):
        raise ValueError(
            "'intents' precisa ser uma lista."
        )

    if not isinstance(dados["desambiguacoes"], list):
        raise ValueError(
            "'desambiguacoes' precisa ser uma lista."
        )

    return dados


# ============================================================
# VALIDAR DATASET
# ============================================================

def validar_dataset(dados):

    print()
    print("=" * 60)
    print("VALIDANDO DATASET")
    print("=" * 60)

    intents = dados["intents"]
    desambiguacoes = dados["desambiguacoes"]

    if not intents:
        raise ValueError(
            "O dataset não possui nenhuma intent."
        )

    nomes_intents = set()

    for i, item in enumerate(intents):

        if not isinstance(item, dict):
            raise ValueError(
                f"Intent #{i} não é um objeto."
            )

        if "intent" not in item:
            raise ValueError(
                f"Intent #{i} não possui a chave 'intent'."
            )

        if "perguntas" not in item:
            raise ValueError(
                f"Intent '{item['intent']}' não possui 'perguntas'."
            )

        if "resposta" not in item:
            raise ValueError(
                f"Intent '{item['intent']}' não possui 'resposta'."
            )

        nome = item["intent"]

        if nome in nomes_intents:
            raise ValueError(
                f"Intent duplicada encontrada: {nome}"
            )

        nomes_intents.add(nome)

        if not isinstance(item["perguntas"], list):
            raise ValueError(
                f"'perguntas' da intent '{nome}' precisa ser uma lista."
            )

        if len(item["perguntas"]) == 0:
            raise ValueError(
                f"A intent '{nome}' não possui perguntas."
            )

        if not isinstance(item["resposta"], str):
            raise ValueError(
                f"'resposta' da intent '{nome}' precisa ser texto."
            )

        for pergunta in item["perguntas"]:

            if not isinstance(pergunta, str):
                raise ValueError(
                    f"Pergunta inválida na intent '{nome}'."
                )

            if not pergunta.strip():
                raise ValueError(
                    f"Pergunta vazia na intent '{nome}'."
                )

    # --------------------------------------------------------
    # VALIDAR DESAMBIGUAÇÕES
    # --------------------------------------------------------

    for i, item in enumerate(desambiguacoes):

        if not isinstance(item, dict):
            raise ValueError(
                f"Desambiguação #{i} não é um objeto."
            )

        obrigatorios = [
            "intents",
            "pergunta",
            "opcoes"
        ]

        for chave in obrigatorios:

            if chave not in item:
                raise ValueError(
                    f"Desambiguação #{i} não possui '{chave}'."
                )

        if not isinstance(item["intents"], list):
            raise ValueError(
                f"'intents' da desambiguação #{i} precisa ser uma lista."
            )

        if len(item["intents"]) != 2:
            raise ValueError(
                f"Desambiguação #{i} deve possuir exatamente 2 intents."
            )

        for nome_intent in item["intents"]:

            if nome_intent not in nomes_intents:
                raise ValueError(
                    f"A desambiguação #{i} referencia "
                    f"a intent inexistente: {nome_intent}"
                )

        if not isinstance(item["opcoes"], list):
            raise ValueError(
                f"'opcoes' da desambiguação #{i} precisa ser uma lista."
            )

        if len(item["opcoes"]) != 2:
            raise ValueError(
                f"Desambiguação #{i} deve possuir exatamente 2 opções."
            )

        for opcao in item["opcoes"]:

            if not isinstance(opcao, dict):
                raise ValueError(
                    f"Opção inválida na desambiguação #{i}."
                )

            if "texto" not in opcao:
                raise ValueError(
                    f"Opção sem 'texto' na desambiguação #{i}."
                )

            if "intent" not in opcao:
                raise ValueError(
                    f"Opção sem 'intent' na desambiguação #{i}."
                )

            if opcao["intent"] not in nomes_intents:
                raise ValueError(
                    f"Opção referencia intent inexistente: "
                    f"{opcao['intent']}"
                )

    print("Estrutura do JSON: OK")
    print(f"Intents válidas: {len(nomes_intents)}")
    print(f"Desambiguações válidas: {len(desambiguacoes)}")


# ============================================================
# PREPARAR PERGUNTAS
# ============================================================

def preparar_dados(dados):

    perguntas = []
    labels = []

    indice_exato = {}

    print()
    print("Preparando perguntas...")

    for item in dados["intents"]:

        intent = item["intent"]

        print(
            f"  - {intent}: "
            f"{len(item['perguntas'])} perguntas"
        )

        for pergunta in item["perguntas"]:

            normalizada = normalizar(pergunta)

            if not normalizada:
                continue

            # Detectar duplicatas
            if normalizada in indice_exato:

                intent_anterior = indice_exato[normalizada]

                if intent_anterior != intent:

                    print()
                    print("ATENÇÃO!")
                    print(
                        "Pergunta duplicada em intents diferentes:"
                    )
                    print(
                        f"Pergunta: {pergunta}"
                    )
                    print(
                        f"1ª intent: {intent_anterior}"
                    )
                    print(
                        f"2ª intent: {intent}"
                    )

            else:

                indice_exato[normalizada] = intent

            perguntas.append(pergunta)
            labels.append(intent)

    return perguntas, labels, indice_exato


# ============================================================
# TESTAR PERGUNTA ESPECÍFICA
# ============================================================

def testar_indice(indice_exato):

    print()
    print("=" * 60)
    print("TESTE DO ÍNDICE")
    print("=" * 60)

    pergunta = "Onde mudo minha senha"

    normalizada = normalizar(pergunta)

    print(f"Pergunta: {normalizada}")

    encontrada = normalizada in indice_exato

    print(f"Encontrada: {encontrada}")

    if encontrada:

        print(
            f"Intent: {indice_exato[normalizada]}"
        )

    else:

        print(
            "ATENÇÃO: pergunta de senha NÃO está no índice!"
        )

        raise ValueError(
            "A pergunta 'Onde mudo minha senha' "
            "não foi encontrada no dataset."
        )


# ============================================================
# TREINAMENTO
# ============================================================

def treinar(perguntas, labels):

    print()
    print("=" * 60)
    print("CRIANDO MODELO TF-IDF + SVM")
    print("=" * 60)

    modelo = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1
            )
        ),
        (
            "svm",
            LinearSVC(
                C=1.5,
                class_weight="balanced"
            )
        )
    ])

    print()
    print("Treinando modelo...")

    modelo.fit(
        perguntas,
        labels
    )

    print()
    print("Modelo treinado com sucesso.")

    return modelo


# ============================================================
# SALVAR MODELO
# ============================================================

def salvar_modelo(
    modelo,
    perguntas,
    labels,
    dados,
    indice_exato
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    pacote = {
        "modelo": modelo,
        "perguntas": perguntas,
        "labels": labels,
        "intents": dados["intents"],
        "desambiguacoes": dados["desambiguacoes"],
        "indice_exato": indice_exato
    }

    joblib.dump(
        pacote,
        MODEL_PATH
    )

    print()
    print("Modelo salvo em:")
    print(MODEL_PATH)


# ============================================================
# VERIFICAR MODELO
# ============================================================

def verificar_modelo():

    print()
    print("=" * 60)
    print("VERIFICANDO MODELO SALVO")
    print("=" * 60)

    pacote = joblib.load(
        MODEL_PATH
    )

    print(
        f"Perguntas no modelo salvo: "
        f"{len(pacote['perguntas'])}"
    )

    pergunta = normalizar(
        "Onde mudo minha senha"
    )

    encontrada = (
        pergunta in pacote["indice_exato"]
    )

    print(
        f"Senha no modelo salvo: "
        f"{encontrada}"
    )

    if not encontrada:

        raise ValueError(
            "ERRO: a pergunta de senha não foi salva no modelo."
        )

    print()
    print("Modelo verificado com sucesso.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        dados = carregar_dataset()

        validar_dataset(
            dados
        )

        perguntas, labels, indice_exato = preparar_dados(
            dados
        )

        print()
        print(
            f"Total de perguntas: {len(perguntas)}"
        )

        print(
            f"Total de intents: "
            f"{len(set(labels))}"
        )

        print(
            f"Total no índice exato: "
            f"{len(indice_exato)}"
        )

        testar_indice(
            indice_exato
        )

        modelo = treinar(
            perguntas,
            labels
        )

        salvar_modelo(
            modelo,
            perguntas,
            labels,
            dados,
            indice_exato
        )

        verificar_modelo()

        print()
        print("=" * 60)
        print("TREINAMENTO FINALIZADO")
        print("=" * 60)

    except Exception as erro:

        print()
        print("=" * 60)
        print("TREINAMENTO INTERROMPIDO")
        print("=" * 60)
        print()
        print(str(erro))
        print()

        raise

