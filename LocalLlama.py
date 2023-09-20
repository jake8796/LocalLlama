# Import modules
from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader, load_index_from_storage
#from llama_hub.file.pdf.base import PDFReader
from llama_cpp import Llama
from pathlib import Path
import os
import time
import argparse
import logging 
import sys


response_modes = {
    "refine": "create and refine an answer by sequentially going through each retrieved text chunk.",
    "compact": "similar to refine but compact (concatenate) the chunks beforehand, resulting in less LLM calls.",
    "tree_summarize": "Query the LLM using the text_qa_template prompt as many times as needed so that all concatenated chunks have been queried, resulting in as many answers that are themselves recursively used as chunks in a tree_summarize LLM call and so on, until there’s only one chunk left, and thus only one final answer.",
    "simple_summarize": "Truncates all text chunks to fit into a single LLM prompt. Good for quick summarization purposes, but may lose detail due to truncation.",
    "no_text": "Only runs the retriever to fetch the nodes that would have been sent to the LLM, without actually sending them. Then can be inspected by checking response.source_nodes.",
    "accumulate": "Given a set of text chunks and the query, apply the query to each text chunk while accumulating the responses into an array. Returns a concatenated string of all responses. Good for when you need to run the same query separately against each text chunk.",
    "compact_accumulate": "The same as accumulate, but will “compact” each LLM prompt similar to compact, and run the same query against each text chunk.",
}

debug_levels = {
    "NotSet" :0,
    "Debug" : 10,
    "Info" : 20,
    "Warning" : 30,
    "Error" :   40,
    "Critical" : 50,
}

# Create a lists for argparse
response_mode_choices = list(response_modes.keys())

debug_level_choices = list(debug_levels.keys())

# Define the command-line arguments and their descriptions
parser = argparse.ArgumentParser(description="PDF Query Script")


# Debug level
parser.add_argument(
    "--debug-level",
    type=str,
    default="NotSet",
    choices=debug_level_choices,
    help="Level of debug prints",
)

# Path to pre-existing embeddings
parser.add_argument(
    "--embedding-path",
    type=str,
    default=None,
    help="Path to pre-existing embeddings for processing or where embeddings from documents will be saved",
)

#Parse PDF path
parser.add_argument(
    "--pdf-folder",
    type=str,
    default="./source",  # Set the default PDF folder path
    help="Path to the folder containing PDFs to parse",
    
)
#Parse Model Path PUT DEFAULT PATH HERE OR IT WON"T WORK
parser.add_argument(
    "--model-path",
    type=str,
    default= r'C:\Users\Jacob\AppData\Local\llama_index\models\llama-2-7b-chat.ggmlv3.q2_K.bin',
    help="Path to the Llama model",
    
)
#Parse Response mode
parser.add_argument(
    "--response-mode",
    type=str,
    choices=response_mode_choices,
    default="compact",  # Set a default response mode
    help=f"Response mode for the query engine. Available modes: {', '.join(response_mode_choices)}",
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
pdf_folder = args.pdf_folder
model_path = args.model_path
usr_response_mode = args.response_mode
embedding_path = args.embedding_path  # Get the embedding path from arguments
debug_level = debug_levels[args.debug_level] #Get debug_level enumeration from dict

# Set the logging level to suppress INFO and DEBUG messages
logging.basicConfig(stream=sys.stdout, level=debug_level)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Set the response mode description based on the chosen response mode
response_mode_description = response_modes.get(usr_response_mode, "Unknown response mode")

print("Resposnse mode chosen: ",response_mode_description)

script_directory = os.path.dirname(os.path.abspath(__file__))
# Specify the model path if specified
if model_path:
    # Specify the model path when initializing LlamaCPP
    llm = Llama(model_path=model_path)

# Change the working directory to the script's directory
os.chdir(script_directory)
print("Current working directory:", os.getcwd())
start_time = time.time()

# Load in files from source directory
documents = SimpleDirectoryReader(pdf_folder).load_data()

# Check if embeddings already exist
if embedding_path and os.path.isfile(os.path.join(embedding_path,"docstore.json")):
    print("Loading embeddings from:", embedding_path)
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
    # load index
    index = load_index_from_storage(storage_context)
else:
    # Process and embed documents
    index = VectorStoreIndex.from_documents(documents)
    # If embeddings path is provided, save the embeddings
    if embedding_path:
        print("Saving embeddings to:", embedding_path)
        # To save embeddings
        index.storage_context.persist(persist_dir=embedding_path)

# Print time to get embedding    
embeddingsTime = time.time()
print("Time taken for embeddings", embeddingsTime - start_time, "seconds\n")

# Retrieval, node poseprocessing, response synthesis. 
query_engine = index.as_query_engine(
    response_mode= usr_response_mode,
    streaming=True, #Makes it so the response prints as it's being generated
    llm_predictor= llm #Changes the model based on the users selection
)

# Get user input for the query
user_query = input("Enter your question: ")

# Run the query engine on the user's question.
response = query_engine.query(user_query)

#Print the response as a stream
response.print_response_stream()

#Print the pages and documents where the information was found
print(response.metadata, "\n")

print("Time taken for response:", (time.time() - embeddingsTime)/60.0, "minutes\n")
print("Total time taken",(time.time() - embeddingsTime)/60.0, "minutes" )
