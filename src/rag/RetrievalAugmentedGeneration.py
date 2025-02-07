################################################################################
#
# Library that makes a RAG model from an Ollama-/Huggingface-based LLM
#
################################################################################

#### TODO
####  * figure out how to threshold similarity in search and suppress low confidence matches
####  * make generalized retriever object and subclass with FAISS and ChromaDB
####    - and try with both vector stores
####    - compare performance of FAISS vs. Chroma for Vector Store

import logging
import numpy as np
import os
from string import Template
import time

#import faiss
import ollama

import pdb  ## pdb.set_trace()  #### TMP TMP TMP

#from rag.retrievers import pdfRetriever, txtRetriever


logger = logging.getLogger(__name__)


class RetrievalAugmentedGeneration():
    def __init__(self, embeddingsStore, modelName, globalContext=""):
        self.embeddingsStore = embeddingsStore
        self.model = modelName
        self.globalContext = globalContext
        self.client = ollama.Client()
        logger.debug("Create RAG")

    def answerQuestion(self, question, numDocs=None, threshold=None):
        # retrieve relevant context from the knowledge base/vector store
        logger.debug("Answer Question")
        metadata = {}
        startTime = time.time()
        context = self.embeddingsStore.getContext(question)
        metadata['getContextTime'] = time.time() - startTime
        metadata['context'] = context

        # generate an answer with the model, using the combined context
        fullPrompt = f"""
Context: {self.globalContext} {context}
Question: {question}
Answer:
"""
        metadata['context'] = context
        startTime = time.time()
        response = self.client.generate(model=self.model, prompt=fullPrompt)
        metadata['generateTime'] = time.time() - startTime
        return response, metadata

if __name__ == "__main__":
    # define whether thoughts are to be printed in answers
    PRINT_THOUGHTS =  True  #False  # True

    K_VAL = 2  # method defaults to 4
    THRESH = None

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
        thoughts, answer = rag.answerQuestion(question)
        if PRINT_THOUGHTS:
            print(f"Thoughts #{i}: {thoughts}")
        print(f"Answer #{i}: {answer}")
        print("--------------------")
