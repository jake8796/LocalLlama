
from llama_index import VectorStoreIndex
from llama_hub.file.pdf.base import PDFReader
from llama_cpp import Llama
from pathlib import Path
import os
import time
import argparse

response_modes = {
    "refine": "create and refine an answer by sequentially going through each retrieved text chunk.",
    "compact": "similar to refine but compact (concatenate) the chunks beforehand, resulting in less LLM calls.",
    "tree_summarize": "Query the LLM using the text_qa_template prompt as many times as needed so that all concatenated chunks have been queried, resulting in as many answers that are themselves recursively used as chunks in a tree_summarize LLM call and so on, until there’s only one chunk left, and thus only one final answer.",
    "simple_summarize": "Truncates all text chunks to fit into a single LLM prompt. Good for quick summarization purposes, but may lose detail due to truncation.",
    "no_text": "Only runs the retriever to fetch the nodes that would have been sent to the LLM, without actually sending them. Then can be inspected by checking response.source_nodes.",
    "accumulate": "Given a set of text chunks and the query, apply the query to each text chunk while accumulating the responses into an array. Returns a concatenated string of all responses. Good for when you need to run the same query separately against each text chunk.",
    "compact_accumulate": "The same as accumulate, but will “compact” each LLM prompt similar to compact, and run the same query against each text chunk.",
}

# Create a list of response mode choices for argparse
response_mode_choices = list(response_modes.keys())

# Define the command-line arguments and their descriptions
parser = argparse.ArgumentParser(description="PDF Query Script")
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

# Set the response mode description based on the chosen response mode
response_mode_description = response_modes.get(usr_response_mode, "Unknown response mode")

print("Resposnse mode chosen: ",response_mode_description)

script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the model path when initializing LlamaCPP
llm = Llama(model_path=model_path)

# Change the working directory to the script's directory
os.chdir(script_directory)
print("Current working directory:", os.getcwd())
start_time = time.time()

# Initialize an empty list to store the loaded documents
all_documents = []

# Iterate through the PDF files in the folder
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        # Load data from the current PDF
        loader = PDFReader()
        documents = loader.load_data(file=Path(pdf_path))
        # Append the loaded documents to the list
        all_documents.extend(documents)

#TODO: Add a way to save embeddings to a specific path then load the embedding instead of processing documents from a specific path every time
#Look at privateGPT for example on how to do this

# Chunking and Embedding of the chunks.
index = VectorStoreIndex.from_documents(documents)

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
print(response.metadata)

print("Time taken for response:", time.time() - start_time, "seconds")
