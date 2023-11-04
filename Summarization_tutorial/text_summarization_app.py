import os
import streamlit as st
from langchain import OpenAI, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
 

st.title("Text Summarizer") 

st.write("check out this [article](https://medium.com/gopenai/extractive-summarization-vs-abstractive-summarization-b60484eb9992)")

def clear_text():
    st.session_state['text'] = ""
    
st.button("clear text input", on_click=clear_text)

OPENAI_KEY = st.text_input("Copy & Paste your OPENAI_API_KEY", type='password') 
if OPENAI_KEY:
 os.environ["OPENAI_API_KEY"] = OPENAI_KEY
 text=st.text_area("Copy & Paste your text data",height=320, key="text")

 tokens = st.slider('Insert a number', max_value=1000)
 st.write('Max number of tokens: ', tokens)

 submit= st.button("Generate Summary")



 llm_model = OpenAI(temperature=0, max_tokens=tokens)
 text_splitter = CharacterTextSplitter()
 split_texts = text_splitter.split_text(text)
 docs = [Document(page_content=t) for t in split_texts]
 chain = load_summarize_chain(llm_model)


 if submit:

     with st.spinner(text="Wait a moment..."):
             response = chain.run(docs)
            
     st.text_area("Output data",value=response, height=320)
    
 def clear_text():
     st.session_state['submit'] = ""
    
 st.button("clear text output", on_click=clear_text)
else:
 st.warning("Please enter your OpenAI API key.")
    
    
