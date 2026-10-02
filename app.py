import os
import re
import unicodedata
import joblib
import json

from flask import Flask, request, jsonify, render_template


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Corrigido para 1 nível
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo.pkl")

FAQ_PATH = os.path.join(
    BASE_DIR,
    "data",
    "faq.json"
)

# ------------------------------------------------------------
# SEGURANÇA DA CLASSIFICAÇÃO
# ------------------------------------------------------------

# Score mínimo absoluto.
#
# Se o melhor score ficar abaixo disso, o chatbot considera
# que não existe confiança suficiente para responder.
#
# Como estamos usando LinearSVC, esse valor não representa
# exatamente uma probabilidade.
#
# Começamos com um valor conservador.
SCORE_MINIMO = 0.10


# Diferença mínima entre o primeiro e o segundo resultado.
#
# Exemplo:
#
# Intent A = 0.50
# Intent B = 0.48
#
# Margem = 0.02
#
# Nesse caso o chatbot NÃO escolhe A.
#
# Ele pede esclarecimento.
MARGEM_MINIMA = 0.25


# Se não existir segundo resultado, podemos aceitar o primeiro
# caso o score seja suficiente.
MARGEM_SEM_SEGUNDO = 999


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)



#rotas

@app.route("/admin", methods=["GET"])
def admin():
    # Renderiza a interface que criaremos no passo 2
    return render_template("admin.html")

@app.route("/api/faq", methods=["GET", "POST"])
def gerenciar_faq():
    # O caminho do FAQ já está definido no seu arquivo como FAQ_PATH
    if request.method == "POST":
        novos_dados = request.get_json()
        
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            # ensure_ascii=False garante que os acentos (ã, é) sejam salvos corretamente
            json.dump(novos_dados, f, indent=2, ensure_ascii=False)
            
        return jsonify({"status": "sucesso", "mensagem": "FAQ atualizado com sucesso!"})

    # Se for GET, apenas retorna o JSON atual
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        dados_faq = json.load(f)
        return jsonify(dados_faq)

# ============================================================
# VARIÁVEIS GLOBAIS
# ============================================================

modelo = None
faq = None

# Índice:
#
# {
#     "pergunta normalizada": "intent"
# }
indice_perguntas = {}

# Respostas:
#
# {
#     "intent": "resposta"
# }
respostas = {}

# Desambiguações:
#
# lista carregada do JSON
desambiguacoes = []

# Guarda a última desambiguação.
#
# Para um projeto simples podemos manter globalmente.
#
# Exemplo:
#
# {
#     "intents": [
#         "visitar_bem",
#         "entrega_bem"
#     ],
#     "opcoes": [...]
# }
ultima_desambiguacao = None


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(texto):
    """
    Normaliza uma pergunta para facilitar comparação.

    Exemplos:

        "Olá!"
        -> "ola"

        "Onde mudo minha senha?"
        -> "onde mudo minha senha"

        "Como faço para dar LANCE?"
        -> "como faco para dar lance"
    """

    if texto is None:
        return ""

    texto = str(texto).strip().lower()

    # Remove acentos
    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    # Remove pontuação
    texto = re.sub(
        r"[^\w\s]",
        " ",
        texto,
        flags=re.UNICODE
    )

    # Remove espaços duplicados
    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto

# ============================================================
# CARREGAR MODELO
# ============================================================

def carregar_modelo():
    global modelo, respostas, desambiguacoes, indice_perguntas

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}. Execute python src/train.py primeiro.")

    print("\n" + "=" * 60)
    print("CARREGANDO MODELO E BASE DE DADOS")
    print("=" * 60)

    # Carrega o pacote inteiro salvo pelo train.py
    pacote = joblib.load(MODEL_PATH)
    
    # Extrai o modelo do dicionário
    modelo = pacote["modelo"]
    
    # Extrai as regras de desambiguação e índice exato diretamente do pacote
    desambiguacoes = pacote.get("desambiguacoes", [])
    indice_perguntas = pacote.get("indice_exato", {})
    
    # Monta o dicionário de respostas de forma limpa
    respostas = {}
    for item in pacote.get("intents", []):
        if "intent" in item and "resposta" in item:
            respostas[item["intent"]] = item["resposta"]

    print("Modelo e contexto carregados com sucesso da memória (.pkl).")


def inicializar():
    print("\n" + "=" * 60)
    print("CHATBOT DE LEILÕES")
    print("=" * 60)

    carregar_modelo()

    print(f"Intents carregadas: {len(respostas)}")
    print(f"Perguntas exatas: {len(indice_perguntas)}")
    print(f"Desambiguações: {len(desambiguacoes)}")


# ============================================================
# MATCH EXATO
# ============================================================

def procurar_match_exato(pergunta_normalizada):

    if pergunta_normalizada in indice_perguntas:

        return indice_perguntas[
            pergunta_normalizada
        ]

    return None


# ============================================================
# PROCURAR DESAMBIGUAÇÃO
# ============================================================

def encontrar_desambiguacao(
    intent1,
    intent2
):

    if not intent1 or not intent2:
        return None

    conjunto_procurado = {
        intent1,
        intent2
    }

    for item in desambiguacoes:

        intents = item.get(
            "intents",
            []
        )

        if len(intents) != 2:
            continue

        if set(intents) == conjunto_procurado:

            return item

    return None


# ============================================================
# CLASSIFICAR COM SVM
# ============================================================

def classificar(pergunta):

    """
    Retorna:

    {
        "intent": "...",
        "score": 0.50,
        "segundo_intent": "...",
        "segundo_score": 0.30,
        "margem": 0.20
    }

    ou None.
    """

    if modelo is None:
        return None

    try:

        # A pipeline TF-IDF + SVM recebe uma lista.
        scores = modelo.decision_function(
            [pergunta]
        )

    except Exception as erro:

        print(
            f"ERRO AO CLASSIFICAR: {erro}"
        )

        return None

    # Algumas versões/configurações podem retornar
    # um array 1D.
    if len(scores.shape) == 1:

        scores = scores.reshape(
            1,
            -1
        )

    scores = scores[0]

    # Classes aprendidas pelo LinearSVC
    classes = modelo.classes_

    resultados = []

    for i, intent in enumerate(classes):

        resultados.append(
            (
                intent,
                float(scores[i])
            )
        )

    # Maior score primeiro
    resultados.sort(
        key=lambda x: x[1],
        reverse=True
    )

    primeiro = resultados[0]

    if len(resultados) > 1:

        segundo = resultados[1]

    else:

        segundo = (
            None,
            0
        )

    intent1 = primeiro[0]
    score1 = primeiro[1]

    intent2 = segundo[0]
    score2 = segundo[1]

    margem = (
        score1 - score2
    )

    return {
        "intent": intent1,
        "score": score1,
        "segundo_intent": intent2,
        "segundo_score": score2,
        "margem": margem,
        "resultados": resultados
    }


# ============================================================
# FALLBACK
# ============================================================

def resposta_fallback():

    return (
        "Não consegui identificar com segurança o que você "
        "deseja saber. Posso ajudar com dúvidas sobre "
        "lances, lotes, arrematação, pagamentos, visitação, "
        "entrega dos bens e outros assuntos relacionados "
        "aos leilões."
    )


# ============================================================
# DESAMBIGUAÇÃO
# ============================================================

def gerar_desambiguacao(
    item
):

    global ultima_desambiguacao

    if not item:
        return None

    pergunta = item.get(
        "pergunta"
    )

    opcoes = item.get(
        "opcoes",
        []
    )

    if not pergunta or not opcoes:
        return None

    ultima_desambiguacao = {
        "intents": item.get(
            "intents",
            []
        ),
        "opcoes": opcoes
    }

    texto = pergunta

    for i, opcao in enumerate(
        opcoes,
        start=1
    ):

        texto += (
            f"\n{i}. "
            f"{opcao.get('texto', '')}"
        )

    return texto


# ============================================================
# PROCESSAR OPÇÃO 1 OU 2
# ============================================================

def processar_opcao(numero):

    global ultima_desambiguacao

    if not ultima_desambiguacao:

        return None

    opcoes = ultima_desambiguacao.get(
        "opcoes",
        []
    )

    if numero < 1 or numero > len(opcoes):

        return (
            "Não consegui identificar sua escolha. "
            "Digite o número da opção desejada."
        )

    opcao = opcoes[
        numero - 1
    ]

    intent = opcao.get(
        "intent"
    )

    # Limpa a desambiguação depois da escolha
    ultima_desambiguacao = None

    if not intent:

        return resposta_fallback()

    resposta = respostas.get(
        intent
    )

    if not resposta:

        return resposta_fallback()

    print()
    print("ESCOLHA DA DESAMBIGUAÇÃO:")
    print(
        f"Opção: {numero}"
    )
    print(
        f"Intent: {intent}"
    )

    return resposta


# ============================================================
# PROCESSAR PERGUNTA
# ============================================================

def processar_pergunta(pergunta):

    global ultima_desambiguacao

    pergunta_original = pergunta

    pergunta_normalizada = normalizar(
        pergunta
    )

    print()
    print("=" * 60)
    print("DEBUG PERGUNTA")
    print(
        f"Original: '{pergunta_original}'"
    )
    print(
        f"Normalizada: '{pergunta_normalizada}'"
    )
    print(
        "Existe no índice:",
        pergunta_normalizada in indice_perguntas
    )
    print("=" * 60)

    # --------------------------------------------------------
    # OPÇÕES DE DESAMBIGUAÇÃO
    # --------------------------------------------------------

    if pergunta_normalizada in (
        "1",
        "2"
    ):

        resultado = processar_opcao(
            int(pergunta_normalizada)
        )

        if resultado:
            return resultado

    # --------------------------------------------------------
    # MATCH EXATO
    # --------------------------------------------------------

    intent_exata = procurar_match_exato(
        pergunta_normalizada
    )

    if intent_exata:

        # Uma nova pergunta normal deve cancelar
        # uma desambiguação anterior.
        ultima_desambiguacao = None

        print()
        print("MATCH EXATO:")
        print(
            f"Pergunta: {pergunta_original}"
        )
        print(
            f"Intent: {intent_exata}"
        )

        resposta = respostas.get(
            intent_exata
        )

        if resposta:
            return resposta

    # --------------------------------------------------------
    # CLASSIFICAÇÃO SVM
    # --------------------------------------------------------

    resultado = classificar(
        pergunta_normalizada
    )

    if not resultado:

        return resposta_fallback()

    intent1 = resultado[
        "intent"
    ]

    score1 = resultado[
        "score"
    ]

    intent2 = resultado[
        "segundo_intent"
    ]

    score2 = resultado[
        "segundo_score"
    ]

    margem = resultado[
        "margem"
    ]

    print()
    print("CLASSIFICAÇÃO:")
    print(
        f"Pergunta: {pergunta_original}"
    )
    print(
        f"1º: {intent1}"
    )
    print(
        f"Score: {score1:.4f}"
    )
    print(
        f"2º: {intent2}"
    )
    print(
        f"Score: {score2:.4f}"
    )
    print(
        f"Margem: {margem:.4f}"
    )

    # --------------------------------------------------------
    # SCORE MUITO BAIXO
    # --------------------------------------------------------

    if score1 < SCORE_MINIMO:

        print()
        print(
            "FALLBACK: score abaixo do mínimo."
        )

        ultima_desambiguacao = None

        return resposta_fallback()

    # --------------------------------------------------------
    # MARGEM PEQUENA
    # --------------------------------------------------------

    if (
        intent2
        and margem < MARGEM_MINIMA
    ):

        print()
        print(
            "POSSÍVEL AMBIGUIDADE."
        )

        print(
            f"Margem {margem:.4f} "
            f"< mínimo {MARGEM_MINIMA:.4f}"
        )

        desambiguacao = encontrar_desambiguacao(
            intent1,
            intent2
        )

        if desambiguacao:

            print(
                "Desambiguação encontrada no FAQ."
            )

            resposta = gerar_desambiguacao(
                desambiguacao
            )

            if resposta:
                return resposta

        # ----------------------------------------------------
        # NÃO EXISTE DESAMBIGUAÇÃO CADASTRADA
        # ----------------------------------------------------

        print(
            "Nenhuma desambiguação cadastrada "
            "para esse par de intents."
        )

        ultima_desambiguacao = None

        return (
            "Fiquei em dúvida entre duas possibilidades. "
            "Você pode reformular sua pergunta com um pouco "
            "mais de detalhes?"
        )

    # --------------------------------------------------------
    # INTENT SEGURA
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICAÇÃO ACEITA."
    )

    print(
        f"Intent escolhida: {intent1}"
    )

    ultima_desambiguacao = None

    resposta = respostas.get(
        intent1
    )

    if not resposta:

        print(
            "ERRO: intent sem resposta no FAQ."
        )

        return resposta_fallback()

    return resposta


# ============================================================
# ROTA PRINCIPAL
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    try:

        return render_template(
            "index.html"
        )

    except Exception:

        return """
        <html>
        <head>
            <title>Chatbot de Leilões</title>
        </head>

        <body>

            <h1>Chatbot de Leilões</h1>

            <p>
                O servidor está funcionando.
            </p>

            <p>
                Acesse a interface do chatbot.
            </p>

        </body>
        </html>
        """


# ============================================================
# API DO CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        dados = request.get_json(
            silent=True
        )

        if not dados:

            return jsonify({
                "erro": "JSON inválido."
            }), 400

        pergunta = dados.get(
            "mensagem"
        )

        # Aceita também "message"
        # caso seu front esteja usando esse nome.
        if not pergunta:

            pergunta = dados.get(
                "message"
            )

        if not pergunta:

            return jsonify({
                "erro": "Mensagem não informada."
            }), 400

        pergunta = str(
            pergunta
        ).strip()

        if not pergunta:

            return jsonify({
                "erro": "Mensagem vazia."
            }), 400

        resposta = processar_pergunta(
            pergunta
        )

        return jsonify({
            "resposta": resposta
        })

    except Exception as erro:

        print()
        print("=" * 60)
        print("ERRO NO /CHAT")
        print("=" * 60)
        print(
            str(erro)
        )

        return jsonify({
            "erro": "Ocorreu um erro ao processar sua pergunta."
        }), 500


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":

    try:

        inicializar()

        print()
        print("=" * 60)
        print("SERVIDOR INICIADO")
        print("=" * 60)
        print(
            "http://127.0.0.1:5000"
        )
        print()

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=True
        )

    except Exception as erro:

        print()
        print("=" * 60)
        print("ERRO AO INICIAR CHATBOT")
        print("=" * 60)
        print(
            str(erro)
        )
        print()
