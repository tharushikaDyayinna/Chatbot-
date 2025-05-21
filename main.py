import logging
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log that the module is being loaded
logger.info("✅ main.py module loaded and ready.")

# Initialize the LLM model
model = OllamaLLM(model="llama3:latest") 

# Define prompt template
template = """
You are an expert support assistant for the Needlu framework.

Here is some relevant reference material to help you answer questions: {reviews}

Please provide a helpful and accurate answer to the following user question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create a chain: prompt → model
chain = prompt | model


def get_answer(question: str) -> str:
    """Fetch answer for a given question using vector retriever and LLM chain."""
    logger.info(f"📨 Received question: {question}")
    reviews = retriever.invoke(question)
    response = chain.invoke({"reviews": reviews, "question": question})
    logger.info("✅ Answer generated.")
    return response  


# For CLI testing (optional)
if __name__ == "__main__":
    print("✅ main.py running in CLI mode")
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question.lower() == "q":
            break
        print(get_answer(question))
