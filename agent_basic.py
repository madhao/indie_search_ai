import streamlit as st
from streamlit_chat import message
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.document_transformers import DoctranTextTranslator
import asyncio
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings("ignore")


# functions
def translator(text, language='english'):
    documents = [Document(page_content=text)]
    qa_translator = DoctranTextTranslator(language=language, openai_api_model='gpt-4')
    # await qa_translator.atransform_documents(documents)
    translated_document = asyncio.run(qa_translator.atransform_documents(documents))
    return translated_document[0].page_content


def generate_response_agent(prompt, chosen_language):
    llm = OpenAI(temperature=0)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Let's test it out!
    output = agent.run(prompt)
    # print(output)
    translation = translator(output, chosen_language)
    # print(translation)
    return translation


# load env
_ = load_dotenv(find_dotenv())  # read local .env file


# webpage
with st.sidebar:
    language_chosen = st.selectbox('Language',
                                   ('English', 'Hindi', 'Marathi')
                                   )

# Creating the chatbot interface
st.title("Indian language search")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    print(user_input)
    output = generate_response_agent(user_input, language_chosen)
    print(output)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
