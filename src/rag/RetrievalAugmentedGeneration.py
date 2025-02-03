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
####  * write an interactive tools that uses this as a library


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
    def __init__(self, embeddingsStore, retriever, modelName, globalContext):
        self.embeddingsStore = embeddingsStore
        self.retriever = retriever
        self.model = modelName
        self.client = ollama.Client()

    def answerQuestion(self, question):
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
        #### FIXME see if I can do this easier with an f-string
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        fullPrompt = Template("""
$globalContext

Context: $context
Question: $question
Answer:
""").substitute(globalContext=globalContext, context=context, question=question)
    
        # generate an answer with the model, using the combined context
        #### FIXME split into thoughts and answer
        response = self.client.generate(model=self.model, prompt=fullPrompt)
        answer = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL)
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
