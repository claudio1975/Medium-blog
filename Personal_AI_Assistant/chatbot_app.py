from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from llamaapi import LlamaAPI
import streamlit as st

# Sidebar for model selection
with st.sidebar:
    option = st.selectbox(
        'Please select your model',
        ('GPT-3.5-turbo', 'Mixtral 8x7B', 'Llama-3-70B'))
    st.write('You selected:', option)

    # API Key input
    api_key = st.text_input("Please Copy & Paste your API_KEY", key="chatbot_api_key", type="password")

    # Reset button
    if st.button('Reset Conversation'):
        st.session_state["messages"] = []
        st.info("Please change your API_KEY if you change model.")
    
    st.write("check out this [article](https://medium.com/@c.giancaterino/build-your-personal-ai-assistant-with-streamlit-and-llms-2f95c9b00e0b)")

# Title and caption
st.title("💬 AI Chatbot")
st.caption("🚀 Your Personal AI Assistant powered by Streamlit and LLMs")



# Initialize messages if not present in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Chat input
if prompt := st.chat_input():
    if not api_key:
        st.info("Please add your API_KEY to go ahead.")
        st.stop()

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Client initialization based on selected model
    if option == 'Mixtral 8x7B':
        client = MistralClient(api_key=api_key)
        response = client.chat(model="open-mixtral-8x7b", messages=st.session_state.messages)
    elif option == 'GPT-3.5-turbo':
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    elif option == 'Llama-3-70B':
        client = OpenAI(api_key=api_key,base_url="https://api.llama-api.com")
        response = client.chat.completions.create(model="llama3-70b", messages=st.session_state.messages)
    else:
        st.error("Selected model is not supported.")
        st.stop()


    # Process response and update session state
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
