import joblib
from pathlib import Path


# ==========================================
# CAMINHO DO MODELO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"


# ==========================================
# CARREGAR MODELO
# ==========================================

print("Carregando chatbot...")

dados = joblib.load(MODEL_PATH)

modelo = dados["modelo"]
respostas = dados["respostas"]

print("Chatbot carregado com sucesso!")
print("Digite 'sair' para encerrar.")
print()


# ==========================================
# FUNÇÃO DE RESPOSTA
# ==========================================

def responder(pergunta):
    """
    Recebe uma pergunta e retorna a resposta
    correspondente à intenção identificada pelo modelo.
    """

    intent = modelo.predict([pergunta])[0]

    resposta = respostas.get(
        intent,
        "Desculpe, não encontrei uma resposta para essa pergunta."
    )

    return resposta


# ==========================================
# CHAT
# ==========================================

while True:

    pergunta = input("Você: ").strip()

    if not pergunta:
        continue

    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("Chatbot: Até logo!")
        break

    resposta = responder(pergunta)

    print()
    print("Chatbot:", resposta)
    print()
