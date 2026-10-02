from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path


# ==========================================
# CONFIGURAÇÃO
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"


# ==========================================
# CARREGAR MODELO
# ==========================================

print("Carregando modelo...")

dados = joblib.load(MODEL_PATH)

modelo = dados["modelo"]
respostas = dados["respostas"]

print("Modelo carregado com sucesso!")


# ==========================================
# FUNÇÃO DO CHATBOT
# ==========================================

def responder(pergunta):

    intent = modelo.predict([pergunta])[0]

    resposta = respostas.get(
        intent,
        "Desculpe, não encontrei uma resposta para essa pergunta."
    )

    return resposta


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# API DO CHAT
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    dados_recebidos = request.get_json()

    pergunta = dados_recebidos.get("mensagem", "").strip()

    if not pergunta:
        return jsonify({
            "resposta": "Digite uma pergunta."
        })

    resposta = responder(pergunta)

    return jsonify({
        "resposta": resposta
    })


# ==========================================
# INICIAR SERVIDOR
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
