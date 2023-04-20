from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI, PromptTemplate
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

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.header("Vyhledávejte v našem seznamu AI pomocí Chat-GPT")

openai_api_key = st.text_input('Běžte na stránku [OpenAI](https://platform.openai.com/account/api-keys), přihlašte se a zkopírujte API klíč:', type='password')
os.environ["OPENAI_API_KEY"] = openai_api_key

model_names = ['text-ada-001', 'text-curie-001', 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4']
selected_model = st.selectbox('Vyberte si model, který chcete, aby náš chat používal (výkonnost modelů je seřazena od shora dolů):', model_names)

st.text("OpenAI dává každému uživateli 18$ k volnému použití. Nemusíte se proto bát, že byste něco platili.")

#template = """Pokud jsi na otázku nenašel odpověď v souboru, prostě řekni "Hmm, Úplně si nejsem jistý." Nesnaž se vymyslet si vlastní odpověď.

#Context: Chováš se jako poradce nebo vyhledávač na stránkách, které obsahují velké množství různých umělých inteligencí. Tvým úkolem je návštěvníkovi vždy správně a slušně poradit s jeho problémem. Máš k dispozici celý seznam všech umělých inteligencí, ve kterém budeš vyhledávat. Informace budeš výhradně brát z tohoto souboru. 
#Struktura souboru: nad každou umělou inteligencí je napsaná cesta, jak se k ní návštěvník může dostat. V každé kategorii Cesta jsou umístěny 3 umělé inteligence, které jsou seřazeny od nejlepší po 3 nejlepší. U nějakých kategorií je i složka "Další" kterou také můžeš využívat.
#"""

def answerMe(vectorIndex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
    prompt = st.text_input('Vyhledávejte: ')
    if prompt:
        response = vIndex.query(prompt, response_mode="compact")
        st.write(f"Odpověď: {response}")

answerMe('vectorIndex.json')
