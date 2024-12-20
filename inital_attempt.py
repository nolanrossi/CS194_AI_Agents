import os
import pdfplumber
import base64
from dotenv import load_dotenv
import time



load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

from openai import OpenAI

client = OpenAI(api_key=api_key)

print("===== PHASE 1: ENVIRONMENT SETUP =====")
print("OpenAI API Key:", api_key)
print("Pinecone API Key:", pinecone_api_key)
print("Pinecone Environment:", pinecone_env)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Pinecone setup
from pinecone import Pinecone, ServerlessSpec

# -------------------------------------
# Helper functions
# -------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------------------------
# PDF Extraction and Preparation
# -------------------------------------
print("===== PHASE 2: PDF EXTRACTION =====")
pdf_file_path = "textbook.pdf"
all_text = ""
with pdfplumber.open(pdf_file_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"

print("PDF text extracted successfully. Length of text:", len(all_text))

def chunk_text(text, max_token_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_token_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

chunks = chunk_text(all_text, max_token_length=500)
print(f"Text has been chunked into {len(chunks)} segments.")

# -------------------------------------
# Embeddings (BERT-based)
# -------------------------------------
print("===== PHASE 3: EMBEDDINGS =====")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")
print("HuggingFace embeddings model loaded successfully.")

# -------------------------------------
# Initialize Pinecone & Build VectorStore
# -------------------------------------


print("===== PHASE 4: PINECONE SETUP =====")


pc = Pinecone(api_key=pinecone_api_key)


index_name = "pinecone-eye-textbook"
dimension = 768  
metric = "cosine"
namespace = "example-namespace"


existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    print(f"Index '{index_name}' not found. Creating...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud='aws', region=pinecone_env)
    )
    print(f"Index '{index_name}' created successfully.")


print("Waiting for the index to be ready...")
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
print("Index is ready.")


index = pc.Index(index_name)


print("Preparing documents for upsert...")
docs = [Document(page_content=chunk) for chunk in chunks]
embeddings = hf_embeddings.embed_documents([doc.page_content for doc in docs])


records = [
    {
        "id": f"doc_{i}",
        "values": embeddings[i],
        "metadata": {"text": doc.page_content}
    }
    for i, doc in enumerate(docs)
]


print("Upserting documents into Pinecone...")
index.upsert(vectors=records, namespace=namespace)
print("Documents successfully upserted into Pinecone.")


time.sleep(5)  
print("Index stats:", index.describe_index_stats())

# -------------------------------------
# Query Enhancement Steps
# -------------------------------------
print("===== PHASE 5: QUERY ENHANCEMENT SETUP =====")
routing_prompt = PromptTemplate(
    input_variables=["user_query"],
    template=(
        "You are a system that decides what the user is really asking. "
        "Given the user query below, summarize it in one sentence focusing "
        "on the key medical/ophthalmological concepts:\n\n"
        "{user_query}\n\n"
        "Summarized Query:"
    )
)

rewriting_prompt = PromptTemplate(
    input_variables=["summarized_query"],
    template=(
        "Rewrite the following query to add pertinent information, "
        "contextual details, and make it more specific for deeper retrieval, "
        "without losing its original meaning:\n\n"
        "{summarized_query}\n\n"
        "Enhanced Query:"
    )
)


llm = ChatOpenAI(
    model="gpt-4o",  
    temperature=0.0,
    openai_api_key=api_key
)

routing_chain = LLMChain(llm=llm, prompt=routing_prompt)
rewriting_chain = LLMChain(llm=llm, prompt=rewriting_prompt)

query_enhancement_chain = SequentialChain(
    chains=[routing_chain, rewriting_chain],
    input_variables=["user_query"],
    output_variables=["enhanced_query"]
)
print("Query enhancement chains set up successfully.")

# -------------------------------------
# RAG Querying Function
# -------------------------------------
def rag_query(query_text: str) -> str:
    print("===== PHASE 6: RAG QUERYING =====")
    print("Original Query:", query_text)
    result = query_enhancement_chain({"user_query": query_text})
    enhanced_query = result["enhanced_query"]
    print("Enhanced Query:", enhanced_query)

    retrieved_docs = retriever.get_relevant_documents(enhanced_query)
    print(f"Retrieved {len(retrieved_docs)} relevant documents from Pinecone.")

    relevant_paragraphs = [doc.page_content.strip() for doc in retrieved_docs if doc.page_content]

    if not relevant_paragraphs:
        return "No relevant information found."
    combined = "\n\n".join(relevant_paragraphs)
    print("RAG query completed successfully.")
    return combined

def get_image_description_from_gpt(image_path: str) -> str:
    print("===== PHASE 7: IMAGE DESCRIPTION GENERATION =====")
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "What is in this image?"
            },
            {
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ],
        temperature=0.2
    )
  
    image_description = response.choices[0].message.content
    print("Image description obtained successfully.")
    return image_description

def get_disease_probabilities() -> str:
    print("===== PHASE 8: DISEASE PROBABILITIES =====")
    nn_output = {
        "a": 0.1,
        "b": 0.05,
        "c": 0.7,
        "d": 0.05,
        "e": 0.05,
        "f": 0.05
    }
    probs_str = "\n".join([f"{d}: {p}" for d, p in nn_output.items()])
    print("Disease probabilities generated successfully.")
    return probs_str

def final_decision(image_description: str, relevant_paragraphs: str, probabilities_str: str) -> str:
    print("===== PHASE 9: FINAL DECISION =====")
    final_prompt = (
        "You are a doctor nerd. "
        "Here is an image of an eyeball and this might be some relevant information from a textbook:\n"
        f"{relevant_paragraphs}\n\n"
        "We also have a state-of-the-art machine learning model, and this is what it says the probabilities "
        "of each disease are:\n"
        f"{probabilities_str}\n\n"
        "Given all your years of being a doctor, what do you think? "
        "Format: Keep it one word / from 1-6 or a-f."
    )

    messages = [
        {"role": "system", "content": "You are a highly knowledgeable ophthalmologist."},
        {"role": "user", "content": final_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    decision = response.choices[0].message.content.strip()
    print("Final decision obtained successfully.")
    return decision

def query_knowledge_base(query_text: str) -> str:
    print("===== PHASE 10: QUERY KNOWLEDGE BASE =====")
    return rag_query(query_text)

if __name__ == "__main__":
    print("===== PHASE 11: MAIN EXECUTION =====")
    image_path = "eye.jpg"  # Local image file
    image_description = get_image_description_from_gpt(image_path)
    relevant_paragraphs = query_knowledge_base(image_description)
    probabilities_str = get_disease_probabilities()
    decision = final_decision(image_description, relevant_paragraphs, probabilities_str)
    print("Final decision:", decision)
    print("===== SCRIPT COMPLETED SUCCESSFULLY =====")
