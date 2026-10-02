# Chatbot Inteligente para Atendimento de Leilões

Chatbot desenvolvido em Python para atendimento automatizado de dúvidas relacionadas a leilões judiciais.

O projeto utiliza técnicas de Machine Learning para classificar a intenção da pergunta do usuário e retornar uma resposta correspondente ao FAQ.

## Tecnologias utilizadas

- Python
- Flask
- Scikit-learn
- TF-IDF
- SVM (LinearSVC)
- Joblib
- JSON
- HTML
- CSS
- JavaScript

## Como funciona

O chatbot utiliza classificação de intenções.

O fluxo básico é:

Usuário
   ↓
Pergunta
   ↓
TF-IDF
   ↓
SVM
   ↓
Identificação da intenção
   ↓
Busca da resposta
   ↓
Resposta ao usuário

Por exemplo:

"Como funciona o lance automático?"

A pergunta é transformada em características pelo TF-IDF.

O classificador SVM identifica a intenção:

lance_automatico

O sistema então busca a resposta associada a essa intenção no dataset.

## Estrutura do projeto

chatbot_leilao/
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
├── README.md
│
├── data/
│   └── faq.json
│
├── models/
│   └── modelo.pkl
│
├── src/
│   ├── train.py
│   └── chatbot.py
│
└── templates/
    └── index.html

## Dataset

O dataset está localizado em:

data/faq.json

O arquivo utiliza uma estrutura baseada em intenções.

Exemplo:

{
    "intent": "lance_automatico",
    "perguntas": [
        "Como funciona o lance automático?",
        "O que é lance automático?",
        "Posso deixar um lance automático?"
    ],
    "resposta": "O lance automático permite programar um incremento fixo e um valor máximo de ofertas..."
}

Cada intenção possui:

- Nome da intenção
- Várias perguntas de treinamento
- Uma resposta associada

A utilização de várias variações de uma mesma pergunta ajuda o modelo a reconhecer diferentes formas de o usuário fazer uma pergunta.

## Instalação

É recomendado utilizar um ambiente virtual.

No Windows:

python -m venv venv

Ative o ambiente:

venv\Scripts\activate

Depois instale as dependências:

pip install -r requirements.txt

## Treinando o modelo

Antes de iniciar o chatbot, é necessário treinar o modelo.

Execute:

python src/train.py

O treinamento irá:

1. Carregar o arquivo data/faq.json
2. Extrair as perguntas
3. Associar cada pergunta à sua intenção
4. Criar o pipeline TF-IDF + SVM
5. Treinar o classificador
6. Salvar o modelo em models/modelo.pkl

Exemplo de saída:

Carregando dataset...
Total de perguntas: ...
Total de intents: ...

Criando modelo TF-IDF + SVM...
Treinando modelo...

Modelo treinado com sucesso!

## Testando pelo terminal

Depois de treinar o modelo, é possível testar o chatbot diretamente pelo terminal:

python src/chatbot.py

Exemplo:

Você: como funciona o lance automático?

Chatbot: O lance automático permite programar um incremento fixo e um valor máximo de ofertas...

Para encerrar:

sair

## Interface Web

O projeto também possui uma interface web desenvolvida com Flask.

Para iniciar:

python app.py

O servidor será iniciado em:

http://127.0.0.1:5000

Abra esse endereço no navegador.

A interface permite enviar perguntas e receber as respostas do modelo em tempo real.

## API interna

A interface utiliza uma rota HTTP:

POST /chat

Exemplo de requisição:

{
    "mensagem": "Como funciona o lance automático?"
}

Exemplo de resposta:

{
    "resposta": "O lance automático permite programar um incremento fixo e um valor máximo de ofertas..."
}

## Intenções

Atualmente o chatbot trabalha com intenções relacionadas a:

- Visitação de bens
- Suspensão da venda
- Segurança do site
- Entrega do bem arrematado
- Condições de venda e pagamento
- Divulgação de leilões
- Valor mínimo de venda
- Impostos e multas
- Comissão do leiloeiro
- Parcelamento
- Auto de Arrematação
- Inadimplência
- Fraude em arrematação
- Incremento mínimo
- Lance
- Arrematantes conjuntos
- Processo do leilão
- Participação no leilão
- Venda em leilão
- Tipos de bens
- Duração do leilão
- Lotes
- Oferta de lances
- Lance antes do apregoamento
- Lance inicial
- Lance condicional
- Lance recebido condicionalmente
- Indicação de leiloeiro
- Lance automático
- Impedimentos
- Artigo 890 do CPC
- Cadastro
- Edital
- Newsletter
- Atendimento

## Modelo de Machine Learning

O classificador utiliza:

TF-IDF
+
LinearSVC

O TF-IDF transforma as perguntas em uma representação numérica baseada na importância dos termos.

O LinearSVC utiliza essa representação para classificar a pergunta em uma das intenções existentes.

## Observação

O chatbot não utiliza a API do ChatGPT.

Todo o processamento do modelo é realizado localmente utilizando Python e Scikit-learn.

Isso permite executar o projeto sem depender de serviços externos de inteligência artificial.

## Próximos passos

Possíveis melhorias para o projeto:

- Melhorar o reconhecimento de intenções semelhantes
- Implementar score de confiança
- Criar resposta de fallback para perguntas desconhecidas
- Melhorar o tratamento de português brasileiro
- Adicionar histórico da conversa
- Adicionar banco de dados
- Criar painel administrativo para editar o FAQ
- Criar interface web mais completa
- Adicionar autenticação
- Disponibilizar o chatbot em produção
- Integrar futuramente com WhatsApp ou outros canais
- Avaliar a possibilidade de utilizar modelos de linguagem para complementar o classificador

## Autor

Projeto de chatbot para atendimento automatizado relacionado a leilões.
