import json
import sys

def validar_json(caminho_arquivo):
    print(f"Iniciando validação do arquivo: {caminho_arquivo}\n")
    erros = 0
    avisos = 0
    
    # 1. Tentar abrir e ler o arquivo JSON (verifica erros de sintaxe, vírgulas faltando, aspas, etc)
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{caminho_arquivo}' não encontrado.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ ERRO DE SINTAXE: O arquivo JSON está mal formatado.")
        print(f"   Detalhes: Linha {e.lineno}, Coluna {e.colno} (Erro: {e.msg})")
        return

    # 2. Verificar se existe a chave 'intents'
    if 'intents' not in dados:
        print("❌ ERRO ESTRUTURAL: A chave principal 'intents' não foi encontrada.")
        return
    
    if not isinstance(dados['intents'], list):
        print("❌ ERRO ESTRUTURAL: 'intents' deve ser uma lista [ ].")
        return

    # 3. Validar cada bloco de intenção
    intents_vistos = set()
    
    for i, item in enumerate(dados['intents']):
        intent_nome = item.get('intent')
        
        # Checar se a intent tem um nome
        if not intent_nome:
            print(f"❌ ERRO (Item {i}): Chave 'intent' está ausente ou vazia.")
            erros += 1
            continue
            
        # Checar se a intent é duplicada
        if intent_nome in intents_vistos:
            print(f"❌ ERRO DE DUPLICIDADE: A intent '{intent_nome}' já foi declarada antes!")
            erros += 1
        else:
            intents_vistos.add(intent_nome)
            
        # Checar se as perguntas existem e são uma lista válida
        perguntas = item.get('perguntas', [])
        if not perguntas or not isinstance(perguntas, list) or len(perguntas) == 0:
            print(f"⚠️ AVISO: A intent '{intent_nome}' não possui 'perguntas' cadastradas.")
            avisos += 1
            
        # Checar se a resposta existe
        resposta = item.get('resposta')
        if not resposta or not isinstance(resposta, str):
            print(f"⚠️ AVISO: A intent '{intent_nome}' não possui uma 'resposta' de texto válida.")
            avisos += 1

    # 4. Resumo final
    print("\n--- RESUMO DA VALIDAÇÃO ---")
    if erros == 0 and avisos == 0:
        print("✅ SUCESSO! O arquivo JSON está perfeito: sem erros de sintaxe e sem duplicadas.")
    elif erros == 0 and avisos > 0:
        print(f"⚠️️ APROVADO COM RESSALVAS: O JSON é válido, mas tem {avisos} aviso(s) (ex: intents sem pergunta/resposta).")
    else:
        print(f"❌ FALHA: O JSON não pode ser usado. Foram encontrados {erros} erro(s). Corrija e tente novamente.")

if __name__ == "__main__":
    # Verifica se o usuário passou o nome do arquivo no terminal
    if len(sys.argv) < 2:
        print("Como usar: python validar_json.py <nome_do_seu_arquivo.json>")
    else:
        validar_json(sys.argv[1])