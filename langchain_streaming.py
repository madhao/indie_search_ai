import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

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


# Main Form
if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)  
    st.toast("Processing... Please wait...", icon='⏳')
    get_result(user_input)