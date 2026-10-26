#imports de las librerias y APIS necesarias 
import os 
import pypdf
import numpy as np
from google import genai
from dotenv import load_dotenv
import llama_index.core as lm_i
from llama_index.core.node_parser import SentenceSplitter
import ollama

TOP_K = 3
Model_embeding = "qwen3-embedding:0.6b"

load_dotenv()

client = genai.Client()

#funcion que genera la respuesta del chatbot a partir de un prompt y el historial de la conversación
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre documentos. "
    "Responde ÚNICAMENTE con la información del Contexto que se te proporciona. "
    "Si la respuesta no está en el Contexto, di que no tienes información suficiente "
    "para responder. No inventes nada. Responde siempre en español."
)

def get_chatbot_response_gemini(mensaje_usuario: str) -> str:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        store=False,
        system_instruction=SYSTEM_PROMPT,
        input=mensaje_usuario,
    )
    return interaction.steps[-1].content[0].text

def get_chatbot_response_ollama(user_message: str):

    response: ollama.ChatResponse = ollama.chat(
    model='gemma4',
    messages=[
        {
            'role': 'system',
            'content': "vas a responderme al prompt con el contexto que te voy a dar, "
            "si no tienes suficiente informacion para responderme, "
            "me vas a decir que no tienes suficiente informacion para responderme y que no inventes nada",
            },
        {
        'role': 'user',
        'content': user_message,
        },
    ],
    )
    return response.message.content

#carga de los documentos que se van a usar para la busqueda de informacion
documents = lm_i.SimpleDirectoryReader(input_files=["./pdfs/Revista16-e7.pdf"]).load_data()

#carga de los documentos en un solo documento para poder hacer la busqueda de informacion
document = lm_i.Document(text="\n\n".join([doc.text for doc in documents]))

print(len(documents))
print(documents[0].text[:500])
#chunking de los documentos para poder hacer la busqueda de informacion

#configuracion del chunking de los documentos (para Rag basico)
splitter = SentenceSplitter(
    #tamaño de cada chunk, es decir, cuantas palabras tiene cada chunk
    chunk_size=512, 
    
    #esta vble define el solapamiento entre los chunks, es decir, cuantas palabras
    #se repiten entre un chunk y otro
    chunk_overlap=20,
    )
nodes = splitter.get_nodes_from_documents([document])

#convertir a vector embedings
#lo configuramos para que lea los chunks de texto y genere un embedding para cada uno de ellos
texts = [node.text for node in nodes]
#result_document = ollama.embed(model = "qwen3-embedding", input = texts)

result_document= ollama.embed(model= Model_embeding, input=texts)

#result_document.embeddings es una lista que contiene los 
# embeddings de cada chunk de texto
#print(result_document.embeddings)

#pasar los embeddings a un array de numpy para poder hacer la busqueda de informacion
#matriz de indices, cada fila es un vector con sus valores de embedding
matriz_embeddings = np.array(result_document.embeddings)

query = "¿Cuántas palabras debe tener el resumen?"
#convertir a vector embedings
#lo configuramos para que lea la query y genere un embedding para cada uno de ellos
result_query = ollama.embed(model= Model_embeding, input = [query])
vector_query = np.array(result_query.embeddings[0])


#funcion que calcula la similitud coseno entre dos vectores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Top k similitudes

#va guardando las similitudes de cada chunk con la query en una lista y las imprime por pantalla
similarities = []
for i, vector in enumerate(matriz_embeddings):
    similarity = cosine_similarity(vector_query, vector)
    #print(f"Chunk {i} similarity: {similarity}")
    similarities.append(similarity)

contexto = []
top_k_posiciones = np.argsort(similarities)[-TOP_K:][::-1]
for rank, i in enumerate(top_k_posiciones, start=1):
    print(f"\n--- Top {rank} | Chunk {i} | similitud: {similarities[i]:.4f} ---")
    print(nodes[i].text)
    contexto.append(nodes[i].text)

texto_contexto = "\n\n".join(contexto)
mensaje_usuario = f"Contexto:\n{texto_contexto}\n\nPregunta: {query}"

respuesta_llm = get_chatbot_response_gemini(mensaje_usuario)
print(respuesta_llm)