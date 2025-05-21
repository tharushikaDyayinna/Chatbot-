import streamlit as st
from main import get_answer  

st.set_page_config(page_title="Needlu Chatbot", layout="centered")

st.title("Chatbot")
st.markdown("Ask anything about the Needlu framework!")

question = st.text_input("📨 Enter your question below:")

# answer
if question:
    with st.spinner("Thinking..."):
        try:
            answer = get_answer(question)
            st.success("✅ Here's the answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error: {e}")
