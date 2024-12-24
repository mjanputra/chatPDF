# ReadMe for Retrieval-Augmented Generation (RAG) Project

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying PDF documents. It combines vector-based document storage using Chroma, local embeddings, and a large language model (LLM) through Ollama to provide context-aware responses to user queries.

### Features
- **PDF Parsing:** Automatically loads all PDFs from a specified directory.
- **Document Chunking:** Splits documents into manageable chunks for vectorization.
- **Vector Storage:** Uses Chroma to store and manage embeddings for efficient similarity search.
- **Query Handling:** Supports querying the database using a natural language interface.
- **Context-Aware Answers:** Constructs prompts with relevant document context and uses an LLM to generate responses.

---

### Key Libraries Used
- `langchain`
- `langchain_community`
- `langchain_huggingface`
- `langchain_ollama`
- `sentence-transformers`
- `transformers`
- `Chroma`
- `PyPDF2`

### Supported Environment
- Python 3.8+
- Works on macOS, Linux, and Windows

---

## Setup
### 1. Install Dependencies
Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Project Directory
- Place all PDF files in the `./data/` directory.
- Ensure that the `chroma/` folder is empty or create it if it doesn’t exist.

### 3. Run the Database Script
To create the Chroma database from the PDF documents, execute:
```bash
python insert_database.py
```

If you want to reset the database and re-ingest the PDFs, use the `--reset` flag:
```bash
python insert_database.py --reset
```

---

## Querying the Database
You can query the Chroma vector store to retrieve relevant document content:

1. Run the query script with a natural language question:
```bash
python query_data.py "What is the total revenue in the company’s annual report?"
```

2. The script:
   - Retrieves the top 5 most relevant chunks based on similarity search.
   - Constructs a prompt with the retrieved context.
   - Queries the Ollama LLM to generate a response.

3. The output includes:
   - Relevant document chunks.
   - Metadata and similarity scores.
   - The AI’s response, citing sources where applicable.

---

## File Structure
```
project/
|— data/                     # Directory containing all PDF files
|— chroma/                   # Chroma vector database
|— insert_database.py       # Script to process and store documents in Chroma
|— query_data.py            # Script to query the database
|— requirements.txt         # List of Python dependencies
|— README.md                # Documentation for the project
```

---

## Key Components

### insert_database.py
- Loads PDF documents from `./data/`.
- Splits documents into smaller chunks.
- Generates embeddings using Hugging Face models.
- Stores the embeddings in Chroma.

### query_data.py
- Queries the Chroma database using similarity search.
- Constructs a prompt with retrieved context.
- Sends the prompt to Ollama’s LLM to generate a response.

---

## Example Workflow
1. **Load PDFs into the Database**
   ```bash
   python insert_database.py
   ```

2. **Query the Database**
   ```bash
   python query_data.py "What are the key findings in the financial report?"
   ```

### Example Output
```
Result 1:
Content: The company achieved a revenue of $10 million in Q1...
Metadata: {'source': 'financial_report.pdf', 'page': 3}
Score: 0.87
...
Response: The company achieved a revenue of $10 million in Q1. Sources: ['financial_report.pdf:3']
```

---

## Troubleshooting

### Common Issues

#### 1. Dependency Errors
If you encounter import or deprecation warnings, ensure all libraries are up-to-date:
```bash
pip install -U langchain_community langchain_huggingface langchain_ollama transformers sentence-transformers
```

#### 2. Missing Chroma Database
Ensure `insert_database.py` is run before querying the database.

#### 3. Ollama Configuration
Make sure the Ollama model (e.g., `mistral`) is installed and accessible:
```bash
ollama pull mistral
```

#### 4. Memory Issues
For large PDFs, consider increasing chunk size or reducing the number of retrieved chunks.
---

## Future Improvements
- Support for multi-language embeddings.
- Integration with other vector databases (e.g., Pinecone, Weaviate).
- Enhanced evaluation metrics for query responses.
---



