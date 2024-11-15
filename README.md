# Pdf-Summarizer-Langauge-Model

A **Streamlit**-based application that allows users to upload a PDF file, extract its content, and generate an AI-generated summary of the content. The app uses Huggingface’s pre-trained language model, **LaMini-Flan-T5-248M**, for text summarization and **LangChain** for managing the language model pipeline.

## Goal of the Project

The goal of this project is to gain hands-on experience with **Streamlit**, **LangChain**, and **Transformer-based models**. It is not intended for deployment or production use at this stage.

## Features

- **Upload PDF files** for summarization.
- **View the uploaded PDF** directly in the app.
- **Automatically summarize** the uploaded PDF.

## Requirements

- **Python**: Programming language
- **Streamlit**: Front-end user interfaces
- **LangChain**: Framework for LLM
- **Huggingface Transformers**: Library for pre-trained models
- **Pytorch**: Framework for model’s computation
- **Base64**: Encoding method

## How it Works

### Model
The app is using **LaMini-Flan-T5-248M**, a fine-tuned version of `google/flan-t5-base` on the **LaMini-instruction dataset** that contains 2.58M samples for instruction fine-tuning.

The LLM **LaMini-Flan-T5-248M** is installed within the project directory for easy integration and use.

### Tokenizer
LangChain’s T5 Tokenizer is used to load the model’s associated tokenizer for converting text into tokenized inputs. T5ForConditionalGeneration is used to fetch the pre-trained model for text summarization.

### Workflow
Upload a PDF file to the app. The file is processed using PyPDFLoader and splits it into text chunks. The text  is then summarized using the model.
The app uses the Hugging Face pipeline for summarisation and LangChain to integrate the PDF loader and model processing. After the PDF is processed, the app generates a summary of the content


## Summary

This project provides a simple and effective tool for PDF summarization, leveraging powerful language models and efficient tools like Streamlit, LangChain, and Hugging Face Transformers.
