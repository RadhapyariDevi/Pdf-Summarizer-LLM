import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os

#MODEL AND TOKENIZER

checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = torch.float32)

#file loader and preprocessing

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap = 100)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts += text.page_content
    return final_texts

#LLM PIPELINE

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer= tokenizer,
        max_length = 500,
        min_length = 50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


#To Display the pdf

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)




#streamlit code
st.set_page_config(layout='wide', page_title="PDF Summarizatio App")


def main():
    st.title('📄PDF Summarization App using LangChain LLM')

    uploaded_file = st.file_uploader("Upload your PDF",type = ['pdf'])

    if uploaded_file is not None:
        if st.button('Summarize'):
            try:
                col1, col2 = st.columns(2)

                filename = uploaded_file.name.replace(" ", "_").replace("&", "and")
                filepath = "data/" + filename
                
                with open(filepath, 'wb') as temp_file:
                    temp_file.write(uploaded_file.read())

                with col1:
                    st.info("Uploaded PDF file")
                    pdf_viewer = displayPDF(filepath)
            
                with col2:
                    st.info("Summarization is below")
                    summary = llm_pipeline(filepath)
                    st.success(summary)

            except FileNotFoundError:
                st.error("File not found. Please check the file path.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()