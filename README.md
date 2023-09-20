# LocalLlama CLI Tool

LocalLlama is a Command-Line Interface (CLI) tool built using the LlamaIndex framework. It is designed to facilitate the loading of documents into a local LLM (Llama Language Model) and querying it with questions. LocalLlama offers several features, including the ability to customize the response mode of the query engine, specify the LLM model (currently supporting only llama models), and the option to save document data in a docstore.

## Features

LocalLlama CLI Tool provides the following key features:

### 1. Document Loading

LocalLlama enables users to load documents into a local LLM. LocalLlama will load documents under the directory /source relative to the path of the script. When specifying the `--embeddings-path` option, it processes and saves the docstores into the specified path.

### 2. Docstore Support

LocalLlama supports the concept of a "docstore," which is a structured data store containing documents. When loading new documents for the first time, you should specify an embedding path to create a docstore. Loading a docstore is significantly faster than processing and loading documents. 

### 3. Response Mode Customization

Users can customize the response mode of the query engine, allowing for tailored interactions with the local LLM.

### 4. Model Selection

LocalLlama currently supports llama models, and users can specify which model to use for their queries.

## Usage

Here's an example of how to use LocalLlama CLI Tool:

```bash
# Load documents and save embeddings to embeddings path
python localLlama.py --embedding-path <path-to-store-docstore>

# Load docstore from embeddings path
python localLlama.py --embedding-path <path-to-store-docstore>
