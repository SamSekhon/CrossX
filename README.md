# crossX

crossX is a comprehensive toolset designed to aid lawyers in preparation and empower future lawyers to practice cross-examination techniques effectively.

## Project Overview

CrossX is a Streamlit app designed to facilitate the cross-examination process through LLMs. It incorporates PyTesseract for text extraction, and utilizes OpenAI Whisper for text-to-speech and speech-to-text functionalities. Additionally, it employs Faiss vector operations for storing and querying embeddings.

The app features a chatbot built with large language models (LLMs) and a Retrieval-Augmented Generation (RAG) approach, implemented using the LangChain framework. This setup includes a history-aware retriever to enhance context relevance. Users can upload their own interviews or depositions in PDF format, or use the two pre-loaded depositions for practice.

## Design

![Alt text](<data/rag.png>)

## Demo

![Alt text](<data/demo.png>)


## Files and Functionality

- **app.py**: Streamlit-based application for interacting with crossX functionalities.

- **packages.txt**: List of dependencies required for the Streamlit application.

- **read.py**: Implements Optical Character Recognition (OCR) for document analysis, aiding in evidence preparation.

- **transcribe_voice_openai.py**: Module for converting text to speech and speech to text, facilitating the practice and analysis of verbal arguments.

- **vector_store.py**: Includes utilities for processing and analyzing Faiss vectors, useful for audio and visual data analysis in courtroom scenarios.

## Installation

1. Clone the repository:

git clone https://github.com/SamSekhon/crossX.git

cd crossX


2. Install dependencies:

pip install -r requirements.txt


Ensure all dependencies listed in `packages.txt` are installed, especially for Streamlit.

## Usage

1. Start the Streamlit application:

streamlit run app.py


Launch the crossX platform to access and utilize various tools for cross-examination preparation and practice.

2. **OCR Functionality**: Utilize `read.py` to extract text from documents, assisting in evidence preparation.

3. **Speech Analysis**: Use `transcribe_voice_openai.py` for converting speech to text and vice versa, facilitating practice sessions for verbal arguments.

4. **Vector Operations**: Refer to `vector_store.py` for tools and utilities related to vector operations.

## Future Works

In future updates, I plan to implement:

- MultiVector Retriever for Image and Video data
- Ability to serve additional case data (Similar to how exhibitions are presented to witnesses)
- Personality feature designed to replicate or alter the emotional responses.

## Contributing

We welcome contributions to enhance crossX. Please feel free to submit pull requests for improvements or new features that benefit the legal community.
