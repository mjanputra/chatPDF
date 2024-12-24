import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from  insert_database import get_embedding

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an AI assistant. Below is some context relevant to a user’s question. 
Use *only* the context provided to formulate your answer—do not use any outside or prior knowledge. 
If the answer is not in the context, say "I don’t know."

Context:
{context}

---

Question:
{question}

Instructions:
1. Rely exclusively on the context above for your answer.
2. If the requested information is not in the context, respond with "I don’t know."
3. When possible, refer to specific details (e.g., page references or IDs) in the context.

Answer:
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    #Prepare the database
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    for idx, (doc, score) in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Score: {score}")
        print("-----------")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    #print(prompt)
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()