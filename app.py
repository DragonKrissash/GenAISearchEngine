import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_core.tools import BaseTool
import os
from dotenv import load_dotenv
load_dotenv()

class SafeTool(BaseTool):
    name: str
    description: str
    tool: object

    def _run(self, query: str):
        try:
            return self.tool.run(query)
        except Exception as e:
            return f"[SafeTool error]: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
arxiv=ArxivQueryRun()
wiki_wrapper=WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)
ddg=DuckDuckGoSearchRun()

ddg_safe = SafeTool(
    name="duckduckgo_search",
    description="Search the web using DuckDuckGo",
    tool=ddg
)

wiki_safe = SafeTool(
    name="wikipedia_search",
    description="Search Wikipedia",
    tool=wiki
)

arxiv_safe = SafeTool(
    name="arxiv_search",
    description="Search academic papers",
    tool=arxiv
)

st.title('Langchain - Search with title')

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant','content':'Hi! I am a chatbot. How can I help you?'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder='What is machine learning'):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)
    llm=ChatGroq(model='llama-3.3-70b-versatile',streaming=True)
    tools = [ddg_safe,wiki_safe,arxiv_safe]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True,verbose=True)
 
    with st.chat_message('assistant'):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        user_input = st.session_state.messages[-1]["content"]
        response = search_agent.run(user_input,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
