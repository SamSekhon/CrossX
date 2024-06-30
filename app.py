# Package imports
import warnings
import streamlit as st
import os
import sys
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
warnings.filterwarnings("ignore")

# Langchain components
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

# Import code modules
from transcribe_voice_openai import *
from vector_store import *


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

# create LLM var
global llm

# Voice dictionary for voice selection
voice_dict = {
    'Adam': 'alloy',
    'Onyx': 'onyx',
    'Nova': 'nova',
    'Ocean': 'shimmer'
}

def initialize_llm(OPENAI_KEY):
    """
    Initialize the language model (LLM).

    Input:
        OPENAI_KEY (str): OpenAI API key
    
    Output:
        llm (ChatOpenAI): Initialized ChatOpenAI object
    """
    return ChatOpenAI(model_name='gpt-4o', temperature=0, openai_api_key=OPENAI_KEY)

def initialize_memory():
    """
    Initialize the memory buffer for conversation summarization.
    
    Output:
        memory (ConversationSummaryBufferMemory): Initialized ConversationSummaryBufferMemory object
    """
    summarizer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # type: ignore
    return ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=summarizer_llm,
        output_key='answer'
    )

def get_system_prompts():
    """
    Define the system prompts for contextualizing questions and generating QA responses.
    
    Output:
        Tuple (contextualize_q_system_prompt, qa_system_prompt): System prompts for contextualization and QA response generation
    """
    contextualize_q_system_prompt = (
        """Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        Do NOT answer the question, just reformulate and make it more clear if needed, otherwise return it as is."""
    )

    qa_system_prompt = (
        """You are an AI assistant to help lawyers of all levels practice cross examination. 
        Response rules:
        1. You are to respond as the witness
        2. The response should be specific to the question asked. Dont give out information not asked.
        "\n\n"
        context: {context}
        Answer:"""
    )
    return contextualize_q_system_prompt, qa_system_prompt

def create_prompts(contextualize_q_system_prompt, qa_system_prompt):
    """
    Create prompt templates using the provided system prompts.
    
    Input:
        contextualize_q_system_prompt (str): Prompt for contextualizing questions
        qa_system_prompt (str): Prompt for generating QA responses
    
    Output:
        Tuple (contextualize_q_prompt, qa_prompt): Created prompt templates
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return contextualize_q_prompt, qa_prompt

def initialize_question_answer_chain(llm, qa_prompt):
    """
    Initialize the question-answer chain.
    
    Input:
        llm (ChatOpenAI): The language model
        qa_prompt (ChatPromptTemplate): The QA prompt template
    
    Output:
        question_answer_chain (DocumentChain): Initialized DocumentChain object
    """
    return create_stuff_documents_chain(llm, qa_prompt)

def initialize_vectordb():
    """
    Initialize the vector database by allowing the user to select an existing collection or create a new one.
    
    Output:
        vectordb (VectorDatabase or None): Initialized VectorDatabase object or None if not created
    """
    # Define the directory where the FAISS indexes are stored
    index_directory = 'faiss/'

    # List all files in the directory
    index_files = os.listdir(index_directory)

    # Filter the list to only include files that match your index naming pattern
    collection_list = [file for file in index_files]

    # collection_list = [i.name for i in list(chroma_client.list_collections())]
    collection_list.append("Create new collection")

    collection_name_str = st.selectbox('Select a collection or create a new one:', collection_list)

    if collection_name_str == "Create new collection":
        new_collection_name = st.text_input("Enter a new collection name")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if st.button('Submit'):
            if new_collection_name and uploaded_file is not None:
                with open(os.path.join(os.getcwd(), new_collection_name + '.pdf'), 'wb') as f:
                    f.write(uploaded_file.getvalue())

                return create_new_collection_streamlit(collection_name_str=new_collection_name, pdf_file=new_collection_name)
    else:
        return load_vector_store(collection_name_str=collection_name_str)

def run_chatbot(client, llm, vectordb, contextualize_q_prompt, question_answer_chain, voice_key, audio_bytes):
    """
    Run the chatbot, handling user input and generating responses.
    
    Input:
        client (OpenAI): OpenAI client
        vectordb (VectorDatabase): Vector database
        contextualize_q_prompt (ChatPromptTemplate): Prompt for contextualizing questions
        question_answer_chain (DocumentChain): Question-answer chain
        voice_key (str): Selected voice key
    
    Output:
        None
    """
    if vectordb is not None:
        retriever = vectordb.as_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        chat_history = []

        # audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

        # if st.button('Ask Question'):
            transcript = record_and_transcribe(client, audio_bytes)
            user_input = transcript

            ai_msg_dict = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            response = ai_msg_dict["answer"]
            chat_history.extend([HumanMessage(content=user_input), response])

            create_output_speech(client, response, voice=voice_dict[voice_key])

            conversation_history = st.session_state.get('conversation_history', [])
            conversation_history.append(('You', user_input))
            conversation_history.append(('Bot', response))
            st.session_state['conversation_history'] = conversation_history

            for role, text in conversation_history:
                st.markdown(f'**{role}**: {text}')

            audio_base64 = convert_audio_to_base64('speech.wav')
            st.markdown(f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>', unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit app.
    
    Input:
        None
    
    Output:
        None
    """
    st.header('Cross Examination')

    # Add voice key selection
    voice_key = st.selectbox('Select Voice:', list(voice_dict.keys()))

    # Add explanation textbox
    explanation_text = """
    #### This application contains two sample collections:
    
    1. **Sample1_Epstein-GhislaineMaxwell-Deposition**: This sample contains deposition of Ghislaine Maxwell related to the Epstein case.
    
    2. **Sample2_Jan6-AlanKlukowski-Deposition**: This sample contains deposition of Alan Klukowski related to January 6 Capitol events.
    
    """
    
    st.markdown(explanation_text)

    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)

    client = OpenAI()

    llm = initialize_llm(OPENAI_KEY = os.environ["OPENAI_API_KEY"])
    memory = initialize_memory()

    contextualize_q_system_prompt, qa_system_prompt = get_system_prompts()
    contextualize_q_prompt, qa_prompt = create_prompts(contextualize_q_system_prompt, qa_system_prompt)

    question_answer_chain = initialize_question_answer_chain(llm, qa_prompt)

    vectordb = initialize_vectordb()

    run_chatbot(client, llm, vectordb, contextualize_q_prompt, question_answer_chain, voice_key, audio_bytes)

if __name__ == '__main__':
    main()