#imports de las librerias y APIS necesarias 
import os 
import pypdf
import numpy as np
from google import genai
from dotenv import load_dotenv
import llama_index as lm_i

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
splitter = lm_i.SentenceSplitter(
    #tamaño de cada chunk, es decir, cuantas palabras tiene cada chunk
    chunk_size=512, 
    
    #esta vble define el solapamiento entre los chunks, es decir, cuantas palabras
    #se repiten entre un chunk y otro
    chunk_overlap=20,
    )
nodes = splitter.get_nodes_from_documents([documents])

#convertir a vector embedings
result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=["un pdf sobre writing", [node.text for node in nodes]]
)
#result embeddings es un diccionario que contiene los 
# embeddings de cada chunk de texto
print(result.embeddings)

#pasar los embeddings a un array de numpy para poder hacer la busqueda de informacion
vectores = [emb.values for emb in result.embeddings]
matriz = np.array(vectores)