import os
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete
#from lightrag.embedding import openai_embedding

# Set your working directory (should be the same as used in your main app)
WORKING_DIR = "./dickens"  # Adjust this path if necessary

# Ensure the working directory exists
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# Set your OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the LLM model function (same as used in your main app)
llm_model_func = gpt_4o_mini_complete

# Define the embedding function
embedding_dimension = 8192  # For OpenAI embeddings (e.g., 'text-embedding-ada-002')
embedding_func = "text-embedding-3-large"

# Initialize LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

# Path to the new text file to be inserted
NEW_TEXT_FILE = "./BusinessRules2.txt"  # Adjust this path if necessary

# Check if the new text file exists
if not os.path.exists(NEW_TEXT_FILE):
    print(f"Error: The file '{NEW_TEXT_FILE}' does not exist.")
    exit(1)

# Read the new text content
with open(NEW_TEXT_FILE, "r", encoding="utf-8") as f:
    new_text = f.read()

# Insert the new text into the existing LightRAG instance
print("Inserting new text into the LightRAG index...")
rag.insert(new_text)
print("Insertion complete.")

# Optional: Regenerate the knowledge graph if needed
# Uncomment the following lines if you want to regenerate the graph

# print("Regenerating the knowledge graph...")
# rag.generate_graph()
# print("Knowledge graph regeneration complete.")
