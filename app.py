import streamlit as st
from openai import OpenAI

st.title("🤖 Local LLM Chat")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        stream = client.chat.completions.create(
            model="qwen3:4b",
            messages=st.session_state.messages,
            stream=True,
        )
        
        for chunk in stream:
            full_response += (chunk.choices[0].delta.content or "")
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})