import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

#Set streamlit Application
st.set_page_config(page_title="Langchain: Summarise a Youtube video or a Website")
st.title(" Langchain: Summarise a Youtube video or a Website")
st.subheader("Summarise URL")


## Get the groq api key and url key field for summarisation

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value="",type="password")

url = st.text_input("URL", label_visibility="collapsed")

if groq_api_key:
    model = ChatGroq(model='Gemma-7b-It',groq_api_key=groq_api_key)

prompt_template = """
    Provide a summary of the following content in 300-350 words :
    Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarise Content from YT or Website"):
    if not groq_api_key.strip():
        st.error("Please provide groq api key")
    elif not url.strip():
        st.error("Please provide URL to summarise")
    
    elif not validators.url(url):
        st.error("Please provide a valid URL to summarise")

    else:
        try:
            with st.spinner("Waiting.."):
                ##loading the website or yt video data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False)
                
                data = loader.load()

                #Chain for summarisation

                chain = load_summarize_chain(model, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(data)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")

                
