import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

##Arxiv wrapper and query run
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

#Wikipedia wrapper and query run
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

#DuckGoSearch
search=DuckDuckGoSearchRun(name='Search')

st.title('Langchain')

st.sidebar.title('Settings')
api_key=st.sidebar.text_input('Enter Your Groq API key',type='password')

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant',"content":"Hi , I am a ChatBot who interact with web, How Can i Help you?"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
    
if prompt:=st.chat_input(placeholder="Type Your Query"):
    st.session_state.messages.append({"role":'user',"content":prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(api_key=api_key,model='Llama3-8b-8192',streaming=True)
    tools=[search,arxiv,wiki]
    
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    
    with st.chat_message('assistant'):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':"assistant","content":response})
        st.write(response)
    
