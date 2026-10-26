#imports de las librerias y APIS necesarias 
import os 
import pypdf
import numpy as np
from google import genai
from dotenv import load_dotenv
import llama_index.core as lm_i
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

client = genai.Client()
history = [
    {
        "type": "user_input",
        "content": [{"type": "text", "text": """eres un chatbot conversacional
        """}]
    }
]

#funcion que genera la respuesta del chatbot a partir de un prompt y el historial de la conversación
def chatbot_response(prompt : str, history: dict):

    history.append({"type": "user_input",
        "content": [{"type": "text", "text": f"{prompt}"}]})
    
    interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    store=False,
    input=history,
    ) 

    for step in interaction.steps:
        history.append(step.model_dump())


#carga de los documentos que se van a usar para la busqueda de informacion
documents = lm_i.SimpleDirectoryReader(input_files=["./pdfs/Revista16-e7.pdf"]).load_data()

#carga de los documentos en un solo documento para poder hacer la busqueda de informacion
document = lm_i.Document(text="\n\n".join([doc.text for doc in documents]))


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

#convertir a vector embeddings

#----- version sin lotes
#result_document = client.models.embed_content(
#        model="gemini-embedding-001",
#        contents=[node.text for node in nodes],
#        config= genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
#)
#-----

#la API admite como maximo 100 textos por llamada, asi que mandamos los chunks en lotes
textos = [node.text for node in nodes]
vectores = []

TAM_LOTE = 100
for i in range(0, len(textos), TAM_LOTE):
    lote = textos[i : i + TAM_LOTE]
    resultado = client.models.embed_content(
        model="gemini-embedding-001",
        contents=lote,
        config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    #añadimos los vectores de este lote a la lista general (se mantiene el orden de nodes)
    vectores.extend(emb.values for emb in resultado.embeddings)

#matriz de embeddings, cada fila es el vector de un chunk
matriz_embeddings = np.array(vectores)

query = "pregunta por definir sobre la revista 16"
#convertir a vector embedings
#lo configuramos para que lea la query y genere un embedding para cada uno de ellos
result_query = client.models.embed_content(
        model="gemini-embedding-001",
        contents= query,
        config= genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)
vector_query = np.array(result_query.embeddings[0].values)


#funcion que calcula la similitud coseno entre dos vectores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Top k similitudes
#va guardando las similitudes de cada chunk con la query en una lista y las imprime por pantalla
simiraties = []
for i, vector in enumerate(matriz_embeddings):
    similarity = cosine_similarity(vector_query, vector)
    print(f"Chunk {i} similarity: {similarity}")
    simiraties.append(similarity)


