import streamlit as st
from langchain.agents import create_openai_tools_agent,AgentType,initialize_agent
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun #Used to run a query i.e. using the tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper #Wrapper used to run the query
from langchain.callbacks import StreamlitCallbackHandler #To make streamlit interact with agents
import os
from dotenv import load_dotenv
from langchain import hub


api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max=250) #Responsible to fetch wikipedia wrappers
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki) #Used to run the query i.e. using the tool so using inbuilt tool of wikipedia

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max=250) #Responsible to fetch arxiv wrappers
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv) #Used to run the query i.e. using the tool so using inbuilt tool of arxiv

search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain Tool as a search engine with the help of tools and agents")


st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter you Groq API Key", type="password")

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant", "content":"Hi, I am a chatbot who can search information using various tools and agents! How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message('user').write(prompt)

    model = ChatGroq(groq_api_key = api_key, model_name = 'Llama3-8b-8192')

    tools = [search, wiki, arxiv]

    agent = initialize_agent(tools, model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION) #To use the agent

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent.run(st.session_state.messages, callbacks = [st_cb])

        st.session_state.messages.append({'role':'user','content': response})

        st.write(response)

    



