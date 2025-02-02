#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# Script that makes a RAG model from an Ollama-/Huggingface-based LLM
#
################################################################################

#### TODO
####  * figure out how to threshold similarity in search and suppress low confidence matches
####  * make generalized retriever object and subclass with FAISS and ChromaDB
####    - and try with both vector stores
####    - compare performance of FAISS vs. Chroma for Vector Store
####  * write an interactive tools that uses this as a library


import logging
import numpy as np
import os
import re
from string import Template

#import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import ollama

import pdb  #### TMP TMP TMP


LOG_LEVEL = "WARNING"

# name of Ollama LLM to use
DEF_LLM_NAME = "deepseek-r1:1.5b"

DEF_CHUNK_SIZE = 2000
DEF_CHUNK_OVERLAP = 0


class RetrievalAugmentedGenerator():
    # instruct model to respond based only on the retrieved context
    PROMPT_TEMPLATE = Template("""
You are an experienced programmer, speaking to another experienced programmer.
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under 4 sentences.

Context: $context
Question: $question
Answer:
""")

    def __init__(self, docsPath,
                 modelName=DEF_LLM_NAME, chunkSize=DEF_CHUNK_SIZE,
                 chunkOverlap=DEF_CHUNK_OVERLAP):
        self.model = modelName
        self.client = ollama.Client()

        # prepare the context document(s)?
        loader = TextLoader(docsPath)
        self.docs = loader.load()
        textSplitter = CharacterTextSplitter(chunk_size=chunkSize,
                                             chunk_overlap=chunkOverlap)
        self.texts = textSplitter.split_documents(self.docs)

        # set up vector store with doc embeddings
        embeddings = HuggingFaceEmbeddings()  # defaults to sentence-transformers/all-mpnet-base-v2
        self.vectorStore = Chroma.from_documents(self.texts, embeddings)

    def answerQuestion(self, question, kVal=None, printThoughts=True):
        # retrieve relevant context from the knowledge base/vector store
        docs = self.vectorStore.similarity_search(question, k=kVal)
        #### TODO add thresholds
        #### N.B. ChromaDB uses cosine distance, so lower means more similar
        #### THRESH = 0.3
#        results = self.vectorStore.similarity_search_with_score(question, k=kVal)
#        filteredResults = [doc for doc, score in results if score <= THRESH]
        #### alternatively
#        retriever = self.vectorStore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.21, 'k': 5}')
        context = "\n".join([doc.page_content for doc in docs])
    
        # combine contexts and query
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        fullPrompt = RetrievalAugmentedGenerator.PROMPT_TEMPLATE.substitute(context=context, question=question)
    
        # generate an answer with the model, using the combined context
        response = self.client.generate(model=self.model, prompt=fullPrompt)
        if printThoughts:
            answer = response['response']
        else:
            answer = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL)
        return answer

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)

    # define whether thoughts are to be printed in answers
    PRINT_THOUGHTS =  True  #False  # True

    K_VAL = 2  # method defaults to 4

    # path to docs that define the knowledge base
    DOCS_PATH = "/home/jdn/Code/LLM_Experiments/assets/cppTutorial.txt"

    # list of questions to ask
    USER_QUESTIONS = (
        "How do I use a namespace?",
        "How does auto type declaration work?",
        "What is your favorite color?")

    MODEL = "deepseek-r1:8b"

    rag = RetrievalAugmentedGenerator(DOCS_PATH)

    for i, question in enumerate(USER_QUESTIONS):
        print(f"Question #{i}: {question}")
        answer = rag.answerQuestion(question, K_VAL, PRINT_THOUGHTS)
        print(f"Answer #{i}: {answer}")
        print("--------------------")
