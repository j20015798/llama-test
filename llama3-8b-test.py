import os
import pickle
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Paths to cache files
embeddings_cache_path = "cached_ollama_embeddings.pkl"
texts_cache_path = "cached_texts.pkl"

# Initialize Ollama model
llm = Ollama(model='llama3', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
print(f"Initialized model: {llm.model}")

# Function to clear cache files (if needed)
def clear_cache():
    if os.path.exists(embeddings_cache_path):
        os.remove(embeddings_cache_path)
    if os.path.exists(texts_cache_path):
        os.remove(texts_cache_path)

# if want to clear the cache
# clear_cache()

# Load and split new PDF file
pdf_path = input("Enter the path of the new PDF file: ")
loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()

# Initialize text splitter
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
documents = text_splitter.split_documents(docs)

# Check if embeddings cache exists
if os.path.exists(embeddings_cache_path) and os.path.exists(texts_cache_path):
    print("Loading cached embeddings and texts...")
    with open(embeddings_cache_path, 'rb') as f:
        embedding_vectors = pickle.load(f)
    with open(texts_cache_path, 'rb') as f:
        texts = pickle.load(f)
else:
    print("Computing embeddings...")
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model='llama3')

    # Compute embeddings and save to cache
    texts = [doc.page_content for doc in documents]
    embedding_vectors = embeddings.embed_documents(texts)

    with open(embeddings_cache_path, 'wb') as f:
        pickle.dump(embedding_vectors, f)
    with open(texts_cache_path, 'wb') as f:
        pickle.dump(texts, f)

# Use FAISS to create vector database
vectordb = FAISS.from_texts(texts, OllamaEmbeddings(model='llama3'))
retriever = vectordb.as_retriever()

# Set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
    })
    '''
    if 'answer' in response:
        print(response['answer'])
    '''
    print('\n')
    input_text = input('>>> ')
