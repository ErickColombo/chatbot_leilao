from flask import Flask, render_template, request, jsonify

import joblib

from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

app = Flask(__name__)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"


# ============================================================
# CONFIGURAÇÕES DO CHATBOT
# ============================================================

# Se True, o chatbot evita responder quando não estiver seguro.
MODO_SEGURO = True


# Quanto maior esse valor, mais facilmente duas intents
# serão consideradas ambíguas.
#
# Como queremos um comportamento conservador,
# deixamos uma margem relativamente alta.
MARGEM_AMBIGUIDADE = 0.25


# Diferença mínima entre as duas melhores intenções
# para considerar que uma delas está claramente na frente.
#
# Exemplo:
#
# intent A = 1.20
# intent B = 0.50
#
# diferença = 0.70
#
# Neste caso, provavelmente não há ambiguidade.
#
# Já:
#
# intent A = 0.85
# intent B = 0.72
#
# diferença = 0.13
#
# Neste caso, podemos pedir esclarecimento.
MARGEM_CLAREZA = 0.25


# ============================================================
# FALLBACK
# ============================================================

RESPOSTA_FALLBACK = (
    "Não consegui identificar com segurança o que você deseja saber. "
    "Posso ajudar com dúvidas sobre lances, lotes, arrematação, "
    "pagamentos, visitação, entrega dos bens e outros assuntos "
    "relacionados aos leilões."
)


# ============================================================
# CARREGAR MODELO
# ============================================================

print("=" * 60)
print("CHATBOT DE LEILÕES")
print("=" * 60)

print("\nCarregando modelo...")

if not MODEL_PATH.exists():

    print("\nERRO: modelo.pkl não encontrado.")

    print(
        "\nExecute primeiro:"
    )

    print(
        "python src/train.py"
    )

    raise SystemExit(1)


dados_modelo = joblib.load(
    MODEL_PATH
)


modelo = dados_modelo["modelo"]

respostas = dados_modelo["respostas"]

desambiguacoes = dados_modelo.get(
    "desambiguacoes",
    []
)


print("Modelo carregado com sucesso.")

print(
    f"Intents carregadas: {len(respostas)}"
)


# ============================================================
# SESSÕES
# ============================================================

# Estrutura:

# sessoes = {
#
#     "usuario_123": {
#
#         "opcoes": [
#             {
#                 "texto": "...",
#                 "intent": "..."
#             }
#         ]
#
#     }
#
# }

sessoes = {}


# ============================================================
# ENCONTRAR DESAMBIGUAÇÃO
# ============================================================

def encontrar_desambiguacao(
    intent1,
    intent2
):

    conjunto = {
        intent1,
        intent2
    }

    for item in desambiguacoes:

        intents_item = set(
            item.get(
                "intents",
                []
            )
        )

        if conjunto.issubset(
            intents_item
        ):

            return item

    return None


# ============================================================
# OBTER RANKING DAS INTENTS
# ============================================================

def obter_intents(
    pergunta
):

    svm = modelo.named_steps["svm"]

    tfidf = modelo.named_steps["tfidf"]


    vetor = tfidf.transform(
        [pergunta]
    )


    scores = svm.decision_function(
        vetor
    )


    classes = svm.classes_


    # --------------------------------------------------------
    # CLASSIFICAÇÃO BINÁRIA
    # --------------------------------------------------------

    if len(classes) == 2:

        score = scores[0]

        resultados = [
            (
                classes[0],
                -score
            ),
            (
                classes[1],
                score
            )
        ]


    # --------------------------------------------------------
    # CLASSIFICAÇÃO MULTICLASSE
    # --------------------------------------------------------

    else:

        scores = scores[0]

        resultados = list(
            zip(
                classes,
                scores
            )
        )


    # Maior score primeiro

    resultados.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return resultados


# ============================================================
# VERIFICAR ESCOLHA DO USUÁRIO
# ============================================================

def processar_escolha(
    pergunta,
    sessao_id
):

    sessao = sessoes.get(
        sessao_id
    )


    if not sessao:

        return None


    opcoes = sessao.get(
        "opcoes"
    )


    if not opcoes:

        return None


    escolha = pergunta.strip()


    # --------------------------------------------------------
    # USUÁRIO DIGITOU NÚMERO
    # --------------------------------------------------------

    if escolha.isdigit():

        numero = int(
            escolha
        )


        if 1 <= numero <= len(opcoes):

            opcao = opcoes[
                numero - 1
            ]


            intent = opcao[
                "intent"
            ]


            # Remove estado da sessão

            sessoes.pop(
                sessao_id,
                None
            )


            return {
                "tipo": "resposta",
                "intent": intent,
                "resposta": respostas.get(
                    intent,
                    RESPOSTA_FALLBACK
                )
            }


        return {
            "tipo": "erro_escolha",
            "resposta": (
                "Opção inválida. "
                "Escolha uma das opções apresentadas."
            ),
            "opcoes": opcoes
        }


    # --------------------------------------------------------
    # USUÁRIO DIGITOU O TEXTO DA OPÇÃO
    # --------------------------------------------------------

    escolha_normalizada = escolha.lower()


    for opcao in opcoes:

        texto_opcao = opcao[
            "texto"
        ].lower()


        if (
            escolha_normalizada
            in texto_opcao
        ):

            intent = opcao[
                "intent"
            ]


            sessoes.pop(
                sessao_id,
                None
            )


            return {
                "tipo": "resposta",
                "intent": intent,
                "resposta": respostas.get(
                    intent,
                    RESPOSTA_FALLBACK
                )
            }


    return {
        "tipo": "erro_escolha",
        "resposta": (
            "Não consegui identificar sua escolha. "
            "Clique em uma das opções ou digite o número correspondente."
        ),
        "opcoes": opcoes
    }


# ============================================================
# PROCESSAR PERGUNTA
# ============================================================

def processar_pergunta(
    pergunta,
    sessao_id
):

    # --------------------------------------------------------
    # PRIMEIRO:
    # verificar se o usuário estava escolhendo uma opção
    # --------------------------------------------------------

    escolha = processar_escolha(
        pergunta,
        sessao_id
    )


    if escolha is not None:

        return escolha


    # --------------------------------------------------------
    # CLASSIFICAR PERGUNTA
    # --------------------------------------------------------

    resultados = obter_intents(
        pergunta
    )


    if not resultados:

        return {
            "tipo": "fallback",
            "resposta": RESPOSTA_FALLBACK
        }


    # --------------------------------------------------------
    # MELHOR INTENT
    # --------------------------------------------------------

    melhor_intent = resultados[0][0]

    melhor_score = resultados[0][1]


    # --------------------------------------------------------
    # SEGUNDA INTENT
    # --------------------------------------------------------

    if len(resultados) > 1:

        segunda_intent = resultados[1][0]

        segundo_score = resultados[1][1]

    else:

        segunda_intent = None

        segundo_score = None


    # --------------------------------------------------------
    # AMBIGUIDADE
    # --------------------------------------------------------

    if segunda_intent is not None:

        diferenca = (
            melhor_score
            - segundo_score
        )


        # ----------------------------------------------------
        # Se as duas estão próximas
        # ----------------------------------------------------

        if diferenca < MARGEM_AMBIGUIDADE:

            desambiguacao = encontrar_desambiguacao(
                melhor_intent,
                segunda_intent
            )


            # ------------------------------------------------
            # Encontrou configuração específica
            # ------------------------------------------------

            if desambiguacao:

                opcoes = desambiguacao[
                    "opcoes"
                ]


                sessoes[sessao_id] = {
                    "opcoes": opcoes
                }


                return {
                    "tipo": "desambiguacao",

                    "resposta": desambiguacao[
                        "pergunta"
                    ],

                    "opcoes": opcoes
                }


    # --------------------------------------------------------
    # MODO SEGURO
    # --------------------------------------------------------

    if MODO_SEGURO:

        # Se temos uma segunda intent e a primeira
        # não abriu vantagem suficiente, não chutamos.

        if (
            segunda_intent is not None
            and (
                melhor_score
                - segundo_score
            ) < MARGEM_CLAREZA
        ):

            return {
                "tipo": "fallback",
                "resposta": RESPOSTA_FALLBACK
            }


    # --------------------------------------------------------
    # RESPOSTA NORMAL
    # --------------------------------------------------------

    resposta = respostas.get(
        melhor_intent
    )


    if not resposta:

        return {
            "tipo": "fallback",
            "resposta": RESPOSTA_FALLBACK
        }


    return {
        "tipo": "resposta",

        "intent": melhor_intent,

        "resposta": resposta
    }


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


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
                "tipo": "erro",
                "resposta": "Requisição inválida."
            }), 400


        pergunta = dados.get(
            "mensagem",
            ""
        )


        sessao_id = dados.get(
            "sessao_id",
            "usuario"
        )


        if not isinstance(
            pergunta,
            str
        ):

            return jsonify({
                "tipo": "erro",
                "resposta": "Mensagem inválida."
            }), 400


        pergunta = pergunta.strip()


        if not pergunta:

            return jsonify({
                "tipo": "erro",
                "resposta": "Digite uma pergunta."
            }), 400


        resultado = processar_pergunta(
            pergunta,
            sessao_id
        )


        return jsonify(
            resultado
        )


    except Exception as erro:

        print(
            "ERRO:",
            erro
        )


        return jsonify({
            "tipo": "erro",
            "resposta": (
                "Ocorreu um erro ao processar sua mensagem."
            )
        }), 500


# ============================================================
# LIMPAR SESSÃO
# ============================================================

@app.route(
    "/chat/reset",
    methods=["POST"]
)
def reset_chat():

    dados = request.get_json(
        silent=True
    ) or {}


    sessao_id = dados.get(
        "sessao_id",
        "usuario"
    )


    sessoes.pop(
        sessao_id,
        None
    )


    return jsonify({
        "status": "ok"
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return jsonify({
        "status": "ok",
        "modelo": "carregado",
        "intents": len(respostas)
    })


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Servidor iniciado em:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
