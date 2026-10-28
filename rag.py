#imports de las librerias y APIS necesarias 
import os 
import json
import numpy as np
from google import genai
from dotenv import load_dotenv
import llama_index.core as lm_i
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
#modelo de lenguaje local
import ollama
#configuracion de certificados SSL para poder usar la API de Ollama y Gemini en Windows
import truststore
truststore.inject_into_ssl()
#base de datos vectorial 
import chromadb

TOP_K = 3
#version embeding local
#Model_embeding = "qwen3-embedding:0.6b"
#version embeding local de hugging face
Model_embeding = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest"

#creacion del cliente de la base de datos vectorial
chroma_client = chromadb.Client()
chroma_client2 = chromadb.Client()
#creacion de la coleccion de la base de datos vectorial
collection = chroma_client.create_collection(name="embeddings_collection")
collection_sentence_window = chroma_client2.create_collection(name="sentence_window_embeddings_collection")

load_dotenv()

client = genai.Client()

#funcion que genera la respuesta del chatbot a partir de un prompt y el historial de la conversación
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre documentos. "
    "Responde ÚNICAMENTE con la información del Contexto que se te proporciona. "
    "Si la respuesta no está en el Contexto, di que no tienes información suficiente "
    "para responder. No inventes nada. Responde siempre en español."
)
query = "¿Cuántas palabras debe tener el resumen?"
query1 = "Si mi artículo lo firmamos nueve personas, ¿cómo pongo los autores al citarlo?"
query2 = "¿Qué diferencia hay entre lo que recomienda la APA y lo que exige la Revista de Mediación sobre la extensión del resumen?"
query3 = "Enumera todos los tipos de artículo que distingue el manual de la APA"   

PROMPT_ANSWER_RELEVANCE = (
    "Eres un evaluador imparcial de sistemas RAG.\n"
    "Tu tarea es evaluar la ANSWER RELEVANCE: si la respuesta contesta de forma directa "
    "y completa a la pregunta del usuario.\n"
    "No evalúes si la información es verdadera ni de dónde sale; solo si responde a lo que se pregunta.\n\n"
    "Escala: del 0 al 1 según la gravedad de la desviación\n"
    "- 0: la respuesta no contesta a la pregunta o habla de otra cosa.\n"
    "- 0.5: contesta solo en parte, de forma incompleta o poco directa.\n"
    "- 1: contesta de forma completa y directa.\n"
    "Puedes usar cualquier valor intermedio con un decimal (por ejemplo 0.3 o 0.8); "
    "usa los valores anteriores como puntos de referencia.\n"
    "Si la respuesta dice que no tiene información suficiente para responder, puntúa 0.\n\n"
    "Devuelve SOLO un JSON con este formato, sin texto adicional ni bloques de código:\n"
    '{{"explicacion": "<razonamiento breve>", "puntuacion": <número entre 0 y 1>}}\n\n'
    "Pregunta:\n{pregunta}\n\n"
    "Respuesta a evaluar:\n{respuesta}\n"
)

PROMPT_CONTEXT_RELEVANCE = (
    "Eres un evaluador imparcial de sistemas RAG.\n"
    "Tu tarea es evaluar la CONTEXT RELEVANCE: si los fragmentos de contexto recuperados "
    "contienen la información necesaria para responder a la pregunta.\n"
    "No hay respuesta que evaluar: juzga solo la relación entre la pregunta y el contexto.\n\n"
    "Escala:\n"
    "- 0: ningún fragmento es relevante para la pregunta.\n"
    "- 0.5: algún fragmento es relevante, pero la información está incompleta.\n"
    "- 1: el contexto contiene toda la información necesaria para responder.\n\n"
    "Puedes usar cualquier valor intermedio con un decimal (por ejemplo 0.3 o 0.8); "
    "usa los valores anteriores como puntos de referencia.\n"
    "Devuelve SOLO un JSON con este formato, sin texto adicional ni bloques de código:\n"
    '{{"explicacion": "<razonamiento breve>", "puntuacion": <número entre 0 y 1>}}\n\n'
    "Pregunta:\n{pregunta}\n\n"
    "Contexto recuperado:\n{contexto}\n"
)

PROMPT_GROUNDEDNESS = (
    "Eres un evaluador imparcial de sistemas RAG.\n"
    "Tu tarea es evaluar la GROUNDEDNESS: si cada afirmación de la respuesta está respaldada "
    "por el contexto proporcionado.\n"
    "NO uses tu propio conocimiento: una afirmación que sea cierta pero no aparezca en el contexto "
    "cuenta como NO respaldada.\n\n"
    "Escala: del 0 al 1 según la gravedad de la desviación\n"
    "- 0: las afirmaciones principales no están respaldadas por el contexto (información inventada).\n"
    "- 0.5: parte de las afirmaciones están respaldadas y parte no.\n"
    "- 1: todas las afirmaciones están respaldadas por el contexto.\n"
    "Puedes usar cualquier valor intermedio con un decimal (por ejemplo 0.3 o 0.8); "
    "usa los valores anteriores como puntos de referencia.\n"
    "Si la respuesta solo dice que no tiene información suficiente, puntúa 1, porque no inventa nada.\n\n"
    "Devuelve SOLO un JSON con este formato, sin texto adicional ni bloques de código:\n"
    '{{"explicacion": "<razonamiento breve>", "puntuacion": <número entre 0 y 1>}}\n\n'
    "Contexto:\n{contexto}\n\n"
    "Respuesta a evaluar:\n{respuesta}\n"
)

PROMPTS_EVALUACION = {
    "answer_relevance": PROMPT_ANSWER_RELEVANCE,
    "context_relevance": PROMPT_CONTEXT_RELEVANCE,
    "groundedness": PROMPT_GROUNDEDNESS,
}



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

#funcion para añadir embeddings dentro de la coleccion de chromadb
def add_to_chroma_collection(collection, texts, embeddings):
        ids = [f"chunk_{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            
        )

def add_to_chroma_collection_sentence_window(collection, texts, embeddings,windows):
        ids = [f"chunk_{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas= windows            
        )


#funcion que calcula la similitud coseno entre dos vectores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_k_similar_chunks(vector_query, collection, top_k=TOP_K):
        #en chromadb, por defecto esta calculado como la distancia coseno
        resultado = collection.query(
        query_embeddings=[vector_query ],
        n_results=top_k)
        return resultado

def get_top_k_similar_chunks_sentence_window(vector_query, collection, top_k=TOP_K):
        #en chromadb, por defecto esta calculado como la distancia coseno
        resultado = collection.query(
        query_embeddings=[vector_query ],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        )
        # sentence window: devolvemos la ventana de contexto, no solo la frase
        ventanas = [meta["window"] for meta in resultado["metadatas"][0]]
        return ventanas        
        
def get_evaluation_response_gemini(evaluation_prompt: list, query, response, context) -> str:

    for evaluation_type, prompt_template in PROMPTS_EVALUACION.items():
        evaluation_prompt = prompt_template.format(
            pregunta=query,
            respuesta=response,
            contexto=context
        )
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            store=False,
            system_instruction=evaluation_prompt,
            input=mensaje_usuario,
        )
    return interaction.steps[-1].content[0].text

def get_judge_response_gemini(evaluation_prompt: str) -> str:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        store=False,
        input=evaluation_prompt,
        generation_config={"temperature": 0},
    )
    return interaction.steps[-1].content[0].text


def evaluate_rag_triad(query: str, response: str, context: str) -> dict:
    resultados = {}
    for evaluation_type, prompt_template in PROMPTS_EVALUACION.items():
        evaluation_prompt = prompt_template.format(
            pregunta=query,
            respuesta=response,
            contexto=context,
        )
        texto_juez = get_judge_response_gemini(evaluation_prompt)

        # El modelo a veces envuelve el JSON en un bloque ```json ... ```
        texto_limpio = texto_juez.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            resultados[evaluation_type] = json.loads(texto_limpio)
        except json.JSONDecodeError:
            resultados[evaluation_type] = {"explicacion": texto_juez, "puntuacion": None}

    return resultados

#carga de los documentos que se van a usar para la busqueda de informacion
documents = lm_i.SimpleDirectoryReader(input_files=["./pdfs/Revista16-e7.pdf"]).load_data()

#carga de los documentos en un solo documento para poder hacer la busqueda de informacion
document = lm_i.Document(text="\n\n".join([doc.text for doc in documents]))

#comprobacion de la carga de los documentos y visualizacion de un fragmento del primer documento
#print(len(documents))
#print(documents[0].text[:500])
#chunking de los documentos para poder hacer la busqueda de informacion

#RAG BASICO-------------------------------------------------------------------------------------------------------------------------------------
#configuracion del chunking de los documentos (para Rag basico)
splitter = SentenceSplitter(
    #tamaño de cada chunk, es decir, cuantas palabras tiene cada chunk
    chunk_size=512, 
    #esta vble define el solapamiento entre los chunks, es decir, cuantas palabras
    #se repiten entre un chunk y otro
    chunk_overlap=20,
    )
nodes = splitter.get_nodes_from_documents([document])

#convertir a vector embedings #Rag basico 
#lo configuramos para que lea los chunks de texto y genere un embedding para cada uno de ellos
if collection.count() == 0 :
    texts = [node.text for node in nodes]
    #result_document = ollama.embed(model = "qwen3-embedding", input = texts)
    result_document= ollama.embed(model= Model_embeding, input=texts)

    #result_document.embeddings es una lista que contiene los 
    # embeddings de cada chunk de texto
    #print(result_document.embeddings)

    #pasar los embeddings a un array de numpy para poder hacer la busqueda de informacion
    #matriz de indices, cada fila es un vector con sus valores de embedding
    #matriz_embeddings = np.array(result_document.embeddings), deja de ser necesario porque ya se hace en la funcion add_to_chroma_collection

    add_to_chroma_collection(collection, texts, result_document.embeddings)
#-------------------------------------------------------------------------------------------------------------------------------------

#RAG SENTENCE_WINDOW -------------------------------------------------------------------------------------------------------------------------------------
#configuracion para el sentence window retrieval, se hace el chunking de cada 
# frase y se guarda el contexto de las ventanas en los metadatos, se crea una lista de nodos
nodes_parser_sentence_window = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodos_sentence_window = nodes_parser_sentence_window.get_nodes_from_documents([document])

#convertir a vector embedings #Rag sentence window 
#lo configuramos para que lea los chunks de texto y genere un embedding para cada uno de ellos
if collection_sentence_window.count() == 0 :
    texts = [node.text for node in nodos_sentence_window]
    result_document= ollama.embed(model= Model_embeding, input=texts)
    metadatas_windows = [node.metadata for node in nodos_sentence_window]
    add_to_chroma_collection_sentence_window(collection_sentence_window, texts, result_document.embeddings, metadatas_windows)
#-------------------------------------------------------------------------------------------------------------------------------------

#convertir a vector embedings
#lo configuramos para que lea la query y genere un embedding para cada uno de ellos
result_query = ollama.embed(model= Model_embeding, input = [query1])
vector_query = np.array(result_query.embeddings[0])

#Top k similitudes
#va guardando las similitudes de cada chunk con la query en una lista y las imprime por pantalla
#similarities = []
#for i, vector in enumerate(matriz_embeddings):
#    similarity = cosine_similarity(vector_query, vector)
#    #print(f"Chunk {i} similarity: {similarity}")
#    similarities.append(similarity)

#contexto = []
#top_k_posiciones = np.argsort(similarities)[-TOP_K:][::-1]
#for rank, i in enumerate(top_k_posiciones, start=1):
#    print(f"\n--- Top {rank} | Chunk {i} | similitud: {similarities[i]:.4f} ---")
#    print(nodes[i].text)
#    contexto.append(nodes[i].text)



#la siguiente linea de codigo hace la busqueda de los embeddings en la coleccion de chromadb y devuelve los resultados 
#esta configurada con la operacion coseno por lo que no hace falta calcular la similitud coseno como se hace en el bucle anterior, 
resultado = get_top_k_similar_chunks(vector_query, collection, top_k=TOP_K)
texto_contexto_basico = "\n\n".join(resultado['documents'][0])
mensaje_usuario = f"Contexto:\n{texto_contexto_basico}\n\nPregunta: {query1}"
respuesta_llm = get_chatbot_response_gemini(mensaje_usuario)

#Lo mismo de arriba pero con sentence_window, la diferencia es que en este caso se devuelve la ventana de contexto y no solo la frase
resultado_sentence_window = get_top_k_similar_chunks_sentence_window(vector_query, collection_sentence_window, top_k=TOP_K)
texto_contexto_sentence_window = "\n\n".join(resultado_sentence_window)
mensaje_usuario_sentence_window = f"Contexto:\n{texto_contexto_sentence_window}\n\nPregunta: {query1}"
respuesta_llm_sentence_window = get_chatbot_response_gemini(mensaje_usuario_sentence_window)

#texto_contexto = "\n\n".join(contexto)
#mensaje_usuario = f"Contexto:\n{texto_contexto}\n\nPregunta: {query}"


print(f"CONTEXTO BASICO : {texto_contexto_basico}")
print(respuesta_llm)
print("------------------------------------------------------------------------------------------------------------------------------------------------------")
print(f"CONTEXTO DE SENTENCE WINDOW : {texto_contexto_sentence_window}")
print(respuesta_llm_sentence_window)
print("------------------------------------------------------------------------------------------------------------------------------------------------------")
print("EVALUACION DE LA RESPUESTA DEL RAG BASICO")
resultados_evaluacion_rag_basico = evaluate_rag_triad(query1, respuesta_llm, texto_contexto_basico)
print(resultados_evaluacion_rag_basico) 
print("------------------------------------------------------------------------------------------------------------------------------------------------------")
print("EVALUACION DE LA RESPUESTA DEL RAG SENTENCE WINDOW") 
resultados_evaluacion_rag_sentence_window = evaluate_rag_triad(query1, respuesta_llm_sentence_window, texto_contexto_sentence_window)
print(resultados_evaluacion_rag_sentence_window)



