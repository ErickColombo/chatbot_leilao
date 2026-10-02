import json
from pathlib import Path


CAMINHO = Path("data/faq.json")


with open(
    CAMINHO,
    "r",
    encoding="utf-8"
) as arquivo:

    dados = json.load(arquivo)


print("Tipo do JSON:", type(dados))

if isinstance(dados, dict):

    print("Chaves principais:")
    print(list(dados.keys()))

    intents = dados.get("intents", [])

else:

    intents = dados


print()
print("Quantidade de intents:", len(intents))
print()


encontrada = False

for item in intents:

    intent = item.get("intent")

    perguntas = item.get("perguntas", [])

    if intent == "alteracao_dados_cadastrais":

        print("INTENT ENCONTRADA:")
        print(intent)

        print()
        print("Quantidade de perguntas:")
        print(len(perguntas))

        print()
        print("Perguntas:")

        for pergunta in perguntas:

            print(repr(pergunta))

            if "Onde mudo minha senha" in pergunta:

                encontrada = True

                print(
                    ">>> ENCONTREI A PERGUNTA <<<"
                )


print()

print(
    "Pergunta encontrada:",
    encontrada
)
