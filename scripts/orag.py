#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# Script that makes a RAG model from an Ollama-based LLM
#
################################################################################

import logging
import numpy as np
from string import Template

import faiss
from huggingface_hub import HuggingFaceEmbeddings
from ollama import Ollama


LOG_LEVEL = "WARNING"

# name of Ollama model to use
LLM_MODEL = "deepseek-r1:1.5b"

# path to docs that define the knowledge base
DOCS_PATH = "${HOME}/Code/LLM_Experiments/assets/"

# instruct model to respond based only on the retrieved context
PROMPT_TEMPLATE = Template("""
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under 4 sentences.

Context: $context
Question: $question
Answer:
""")

USER_QUESTIONS = (
    "What are the key features of DeepSeek R1?",
    "What is your favorite color?",
    "What is the meaning of life?")


def answerQuery(llm, retriever, propmtTemplate, question):
    # retrieve relevant context from the knowledge base
    context = retriever.retrieve(question)
    
    # combine retrieved contexts into a single string (if multiple)
    combinedContext = "n".join(context)
    
    # generate an answer using the LLM with the combined context
    response = llm.generate(promptTemplate.substitute(context=combinedContext, question=question))
    return response.strip()

class SimpleRetriever():
    def __init__(self, index, embeddingsModel):
        self.index = index
        self.embeddingsModel = embeddingsModel
    
    def retrieve(self, query, k=3):
        queryEmbedding = self.embeddingsModel.embed(query)
        distances, indices = self.index.search(np.array([queryEmbedding]).astype('float32'), k)
        return [documents[i] for i in indices[0]]

class RetrievalAugmentedGenerator():
    def __init__(self, model, promptTemplate):
        self.llm = Ollama(model=model)
        self.promptTemplate = promptTemplate

    def _loadDocuments(directory):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as file:
                    documents.append(file.read())
        return documents

    def setKnowledgeBase(self, docsPath):
        self.docs = self._loadDocuments(docsPath)

        # init the embeddings model and generate embeddings for all documents
        self.embeddingsModel = HuggingFaceEmbeddings()
        docEmbeddings = [embeddingsModel.embed(doc) for doc in self.docs]
        self.docEmbeddings = np.array(docEmbeddings).astype('float32')

        # create FAISS index and add document embeddings to the index
        self.index = faiss.IndexFlatL2(self.docEmbeddings.shape[1])  # use L2 distance metric
        self.index.add(docEmbeddings)

        # create retriever based on user queries to fetch most relevant documents
        self.retriever = SimpleRetriever(self.index, self.embeddingsModel)

    def answerQuestion(self, question):
        # retrieve relevant context from the knowledge base
        context = self.retriever.retrieve(question)
    
        # combine retrieved contexts into a single string (if there are multiple of them)
        combinedContext = "n".join(context)
    
        # generate an answer with the model, using the combined context
        response = self.llm.generate(self.promptTemplate.substitute(context=combinedContext, question=question))
        return response.strip()

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)
    rag = RetrievalAugmentedGenerator(LLM_MODEL, PROMPT_TEMPLATE)
    rag.setKnowledgeBase(DOCS_PATH)

    for i, question in enumerate(USER_QUESTIONS):
        print(f"Question #{i}: {question}")
        answer = rag.answerQuery(question)
        print(f"Answer #{i}: {answer}")

