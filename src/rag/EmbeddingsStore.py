################################################################################
#
# Library that Implements a Base Class for the Document Embeddings Store
#
################################################################################

import logging
from pathlib import Path

import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import unstructured

import pdb  ## pdb.set_trace()  #### TMP TMP TMP


logger = logging.getLogger(__name__)


#### TODO make vector store selectable

class EmbeddingsStore():
    def __init__(self, modelName, chunkSize, chunkOverlap):
        self.vectorStore = None
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.embeddings = HuggingFaceEmbeddings(model_name=modelName)
        self.clientSettings = chromadb.config.Settings(anonymized_telemetry=False)

    # can override for different loader and splitter
    def createStore(self, docsPath, persistPath=None):
        if self.vectorStore:
            raise ValueError("Already using a vector store, must delete it first")
        self.docsPath = Path(docsPath)
        self.persistPath = persistPath
        self.texts = []
        logger.debug(f"Create Embeddings Store: {docsPath}, {self.chunkSize}, {self.chunkOverlap}, {self.threshold}, {self.persistPath}")

        textSplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
                                                      chunk_size=self.chunkSize,
                                                      chunk_overlap=self.chunkOverlap)
        #### TODO Dry this up
        txtDocs = 0
        numPages = 0
        for filePath in self.docsPath.glob("**/*.txt", case_sensitive=False):
            loader = TextLoader(str(filePath))
            pages = loader.load()
            numPages += len(pages)
            self.texts += textSplitter.split_documents(pages)
            txtDocs += 1
        logger.info(f"# Text Docs: {txtDocs}; # Pages: {numPages}; # Texts: {len(self.texts)}")

        pdfDocs = 0
        numPages = 0
        for filePath in self.docsPath.glob("**/*.pdf", case_sensitive=False):
            loader = PyPDFium2Loader(str(filePath))
            pages = loader.load()
            numPages += len(pages)
            self.texts += textSplitter.split_documents(pages)
            pdfDocs += 1
        logger.info(f"Number of pdf Docs: {pdfDocs}; # Pages: {numPages}; # Texts: {len(self.texts)}")
        if not txtDocs and not pdfDocs:
            raise AssertionError("No documents")

        self.vectorStore = Chroma.from_documents(self.texts, embedding=self.embeddings,
                                                 persist_directory=self.persistPath,
                                                 client_settings=self.clientSettings)

    def useStore(self, persistPath):
        logger.debug(f"Use Existing Embeddings Store: {persistPath}")
        if self.vectorStore:
            raise ValueError("Already using a vector store, can't create another one without first doing a delete")
        self.vectorStore = Chroma(embedding_function=self.embeddings,
                                  persist_directory=persistPath,
                                  client_settings=self.clientSettings)

    def deleteStore(self):
        raise Exception("TBD")
        self.vectorStore = None

    #### FIXME allow for different types similarity functions -- e.g., dot and cosine
    def getContext(self, question, maxContext, threshold):
        # retrieve relevant context from the knowledge base/vector store
        results = self.vectorStore.similarity_search_with_score(question, k=(maxContext // self.chunkSize))
        logger.info(f"# Docs retrieved: {len(results)}")
        chunks = [chunk for chunk, score in results]
        titles = {chunk.metadata['source']: sum(1 for item in chunks if item.metadata['source'] == chunk.metadata['source']) for chunk in chunks}
        logger.info(titles)
        scores = [(chunk.metadata['source'], score) for chunk, score in results]
        logger.info(scores)
        pdb.set_trace()  #### TMP TMP TMP
        #### N.B. ChromaDB uses cosine distance, so lower means more similar
        filteredChunks = [chunk for chunk, score in results if score <= threshold]
        #### alternatively
#        retriever = self.vectorStore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.21, 'k': 5}')
        context = "\n".join([chunk.page_content for chunk in filteredChunks])
        logger.info(f"Size of thresholded context: {len(context)} Bytes")
        return context
