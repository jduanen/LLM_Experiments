#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# Script that makes a RAG model from an Ollama-based LLM
#
################################################################################

import logging
import numpy as np
import os
from string import Template

import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

####import pdb  #### TMP TMP TMP


LOG_LEVEL = "WARNING"

# name of Ollama LLM to use
DEF_LLM_NAME = "deepseek-r1:1.5b"

# name of Huggingface Embedding Model to use
DEF_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class SimpleRetriever():
    def __init__(self, documents, index, embeddingModel):
        self.documents = documents
        self.index = index
        self.embeddingModel = embeddingModel

    def retrieve(self, query, k=3):
        queryEmbedding = self.embeddingModel.embed_query(query)
        distances, indices = self.index.search(np.array([queryEmbedding]).astype('float32'), k)
        return [self.documents[i] for i in indices[0]]

class RetrievalAugmentedGenerator():
    # instruct model to respond based only on the retrieved context
    PROMPT_TEMPLATE = Template("""
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under 4 sentences.

Context: $context
Question: $question
Answer:
""")

    @staticmethod
    def _loadDocuments(directory):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as file:
                    documents.append(file.read())
        return documents

    def __init__(self, docsPath,
                 embeddingModelName=DEF_EMBEDDING_MODEL_NAME,
                 modelName=DEF_LLM_NAME):
        self.model = modelName
        self.client = ollama.Client()

        self.docs = RetrievalAugmentedGenerator._loadDocuments(docsPath)

        # init the embedding model and generate embeddings for all documents
        self.embeddingModel = HuggingFaceEmbeddings(model_name=embeddingModelName)
        docEmbeddings = self.embeddingModel.embed_documents(self.docs)
        self.docEmbeddings = np.array(docEmbeddings).astype('float32')

        # create FAISS index and add document embeddings to the index
        self.index = faiss.IndexFlatL2(self.docEmbeddings.shape[1])  # use L2 distance metric
        ####pdb.set_trace()
        self.index.add(self.docEmbeddings)

        # create retriever based on user queries to fetch most relevant documents
        self.retriever = SimpleRetriever(self.docs, self.index, self.embeddingModel)

    def answerQuestion(self, question):
        # retrieve relevant context from the knowledge base
        context = self.retriever.retrieve(question)
    
        # combine retrieved contexts into a single string (if there are multiple of them)
        combinedContext = "n".join(context)
    
        # generate an answer with the model, using the combined context
        fullPrompt = RetrievalAugmentedGenerator.PROMPT_TEMPLATE.substitute(context=combinedContext, question=question)
        print("PROMPT START --------------------")
        print(fullPrompt)
        print("PROMPT END -------------------")
        answer = self.client.generate(model=self.model, prompt=fullPrompt)
        return answer.response

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)

    # path to docs that define the knowledge base
    DOCS_PATH = "/home/jdn/Code/LLM_Experiments/assets/"

    # list of questions to ask
    USER_QUESTIONS = (
        "How do I use a namespace?",
        "How does auto type declaration work?",
        "What is your favorite color?")

    rag = RetrievalAugmentedGenerator(DOCS_PATH)

    for i, question in enumerate(USER_QUESTIONS):
        print(f"Question #{i}: {question}")
        answer = rag.answerQuestion(question)
        print(f"Answer #{i}: {answer}")
        print("--------------------")
