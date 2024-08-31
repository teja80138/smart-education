from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
# from helper_utils import project_embeddings, word_wrap
# Create a loader for your CSV file
loader = CSVLoader("example.csv")

# Load the data
data = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, add_start_index=True)
documents = text_splitter.split_documents(data)

# Extract text from documents
texts = [doc.page_content for doc in documents]

# Initialize the SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()

# Initialize Chroma client and create a collection with the embedding function
# chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    name="testing_collection",
    embedding_function=embedding_function
)

# Prepare IDs for the documents
ids = [str(i) for i in range(len(texts))]

# Add documents to the Chroma collection
chroma_collection.add(ids=ids, documents=texts)

# Query the collection
query = 'total students enrolled for National Institute of Technology Rourkela'
results = chroma_collection.query(query_texts=[query], n_results=5)

# Retrieve the documents
retrieved_documents = results["documents"][0]

# Print results

## Load Ollama LAMA2 LLM model
# llm=ChatOllama(model="llama3")
# prompt={
#     ''''''
# }
def augment_query_generated(query,llm):
    prompt = """You are a helpful assistant that must try your best effort to answer the user question ALWAYS following this guidelines:
                Keep your answer ground in the facts provided in DOCUMENT section.
                If the DOCUMENT section doesn’t contain the facts to answer the QUESTION ALWAYS return {'sorry your question is not relavent. if there are any questions fell free to ask'}."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    chain=contextualize_q_prompt|llm| StrOutputParser()
    return chain.invoke({"input":query})      

original_query="what is indias best university?"


llm=ChatOllama(model="llama3")
hypothetical_answer=augment_query_generated(original_query,llm)

# hypothetical_answer = augment_query_generated(original_query)
joint_query = f"{original_query} {hypothetical_answer}"
# print(word_wrap(hypothetical_answer))
results = chroma_collection.query(
    query_texts=hypothetical_answer, n_results=5, include=["documents", "embeddings"]
)

print("\n\n\n\n\n\n\n\n\n\n\n")
retrieved_documents = results["documents"][0]
print(retrieved_documents[0])
# model=