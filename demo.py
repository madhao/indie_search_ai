import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import openai
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
from app import generate_response_agent

model = ChatOpenAI(
                   streaming=True,
                   callbacks=[StreamingStdOutCallbackHandler()],
                   verbose=True)
st.title("Indian language search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_result(user_input):
    message_system = SystemMessage(content="You're are a helpful," 
                                          "talkative, and friendly assistant.")
    # Prepare the user message using the value from the `text_area`
    message_user = HumanMessage(content=user_input)
    full_response = []
    # Loop through the chunks streamed back from the API call
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for response in model.stream([message_system, message_user]): 
            wordstream = response.dict().get('content')
            if wordstream:
                full_response.append(wordstream)
                result = "".join(full_response).strip()
                # print(full_response)
            result = "".join(full_response).strip()
            message_placeholder.markdown(result + "▌")
        st.session_state.messages.append({"role": "assistant", "content": result})


def get_result_agent(user_input):
    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    llm = ChatOpenAI(temperature=0, callbacks=[FinalStreamingStdOutCallbackHandler()],streaming=True)
    tools = load_tools(["serpapi", "llm-math","wikipedia"], llm= llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Let's test it out!
    # print(output)
    full_response = []
    # Loop through the chunks streamed back from the API call
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for response in agent.stream(user_input):
            print(response)
            wordstream = response.get('output')
            if wordstream:
                full_response.append(wordstream)
                result = "".join(full_response).strip()
                # print(full_response)
            result = "".join(full_response).strip()
            message_placeholder.markdown(result + "▌")
        st.session_state.messages.append({"role": "assistant", "content": result})


# Main Form
if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)  
    st.toast("Processing... Please wait...", icon='⏳')
    get_result_agent(user_input)