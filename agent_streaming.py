from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text=""):
        self.container = st.empty()
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# class StreamHandler(BaseCallbackHandler):
#     def __init__(self):
#         self.msg = cl.Message(content="")

#     async def on_llm_new_token(self, token: str, **kwargs):
#         await self.msg.stream_token(token)

#     async def on_llm_end(self, response: str, **kwargs):
#         await self.msg.send()
#         self.msg = cl.Message(content="")


with st.sidebar:
    language_chosen = st.selectbox('Language',
                                   ('English', 'Hindi', 'Marathi')
                                   )

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    message_system = SystemMessage(content="You're are a helpful,"
                                           "talkative, and friendly assistant."
                                           "Translate the output to {}".format(language_chosen)
                                           )
    message_user = HumanMessage(content=prompt)
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler()
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler],temperature=0,verbose=True)
        tools = load_tools(["serpapi", "llm-math","wikipedia"], llm= llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        response = llm([message_system, message_user])
        print(response)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))