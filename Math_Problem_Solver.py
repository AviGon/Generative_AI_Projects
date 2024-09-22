import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up the Stramlit app
st.set_page_config(page_title="Text To Math Problem Solver",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq API Key",value="",type="password")

if not groq_api_key:
    st.error("Please add your Groq API key to continue")

if groq_api_key:
    model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## Initialize the Math tool
    math_chain=LLMMathChain.from_llm(llm=model)

    calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided"
    )

    prompt="""
    You are a mathematical agent to help people solve their mathematical queries. Please understand the question and provide the answer but with
    logical reasoning and the complete step by step approach to solving the problem. Please provide a detailed explanation.
    Incase the details given in the question is irrevelant to the question asked, please ask for more information but do not give information based on your knowledge or assume things.
    Question:{question}
    Answer: Given you answer here
    """

    prompt_template=PromptTemplate(
        input_variables=["question"],
        template=prompt
    )

## Combine all the tools into chain
    chain=LLMChain(llm=model, prompt=prompt_template)

    reasoning_tool=Tool(
        name="Reasoning tool",
        func=chain.run,
        description="A tool for answering logic-based and reasoning questions."
    )

## Initialize the agent

    assistant_agent=initialize_agent(
        tools=[calculator,reasoning_tool],
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #Reason before answering agent
        verbose=False,
        handle_parsing_errors=True
    )

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths questions"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

## Lets start the interaction
    question=st.text_area("Enter your question:")

    if st.button("Solve"):
        if question:
            try:
                with st.spinner("Generating.."):
                    st.session_state.messages.append({"role":"user","content":question})
                    st.chat_message("user").write(question)

                    st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                    response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
                    st.session_state.messages.append({'role':'assistant',"content":response})
                    st.write('The Response is:')
                    st.success(response)
            
            except Exception as e:
                st.exception(f"Exception: {e}")

        else:
            st.error("Please enter the question")
