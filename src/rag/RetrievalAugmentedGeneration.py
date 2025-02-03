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
import re
from string import Template

#import faiss
import ollama

import pdb  #### TMP TMP TMP

#from rag.retrievers import pdfRetriever, txtRetriever


class RetrievalAugmentedGeneration():
    def __init__(self, embeddingsStore, modelName, globalContext=""):
        self.embeddingsStore = embeddingsStore
        self.model = modelName
        self.globalContext = globalContext
        self.client = ollama.Client()

    def answerQuestion(self, question, numDocs=None, threshold=None):
        # retrieve relevant context from the knowledge base/vector store
        context = self.embeddingsStore.getContext(question)
    
        # combine contexts and query
        #### FIXME see if I can do this easier with an f-string
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        fullPrompt = Template("""
$gcontext

Context: $context
Question: $question
Answer:
""").substitute(gcontext=self.globalContext, context=context, question=question)
    
        # generate an answer with the model, using the combined context
        response = self.client.generate(model=self.model, prompt=fullPrompt)
        pattern = f"({re.escape('<think>')}.*?{re.escape('</think>')})"
        parts = re.split(pattern, response['response'], maxsplit=1, flags=re.DOTALL)
        if len(parts) == 3:
            thoughts = parts[1]
            answer = parts[2]
        elif len(parts) == 1:
            thoughts = None
            answer = parts[0]
        else:
            logging.error(f"Confused response: {parts}")
            thoughts = None
            answer = None
        return thoughts, answer

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)

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
