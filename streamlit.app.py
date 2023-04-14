from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
import streamlit as st

# Define some HTML with a div element with class "hide-me"
html = """
<div class="stException">
  This element should be hidden
</div>
"""

# Define the CSS to hide elements with class "hide-me"
hide_css = """
.stException {
    display: none;
}
"""

# Use st.markdown to render the HTML and the CSS
st.markdown(html, unsafe_allow_html=True)
st.write(f'<style>{hide_css}</style>', unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.header("Vyhledávejte v našem seznamu AI pomocí Chat-GPT")

openai_api_key = st.text_input('Běžte na stránku [OpenAI](https://platform.openai.com/account/api-keys), přihlašte se a zkopírujte API klíč:', type='password')
os.environ["OPENAI_API_KEY"] = openai_api_key

model_names = ['text-ada-001', 'text-curie-001', 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4']
selected_model = st.selectbox('Vyberte si model, který chcete, aby náš chat používal (výkonnost modelů je seřazena od shora dolů):', model_names)

st.text("OpenAI dává každému uživateli 18$ k volnému použití. Nemusíte se proto bát, že byste něco platili.")

def createVectorIndex(path):
    max_input = 8192
    tokens = 2000
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=selected_model, maximum_tokens=2000))

    docs = SimpleDirectoryReader(path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=PromptHelper)
    vectorIndex = GPTSimpleVectorIndex.from_documents(documents=docs,service_context=service_context)

    vectorIndex.save_to_disk('vectorIndex.json')

    return vectorIndex

def answerMe(vectorIndex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
    prompt = st.text_input('Co máš na srdíčku: ')
    if prompt:
        response = vIndex.query(prompt, response_mode="compact")
        st.write(f"Odpověď: {response}")

vectorIndex = createVectorIndex('data123')

answerMe('vectorIndex.json')
