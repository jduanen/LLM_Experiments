################################################################################
#
# Library that Implements a Base Class for the Document Embeddings Store
#
################################################################################

import json
import logging
import os
from pathlib import Path
import shutil

import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import unstructured

import pdb  ## pdb.set_trace()  #### TMP TMP TMP


logger = logging.getLogger(__name__)


#### FIXME make vector store selectable
class EmbeddingsStore():
    def __init__(self, modelName, chunkSize, chunkOverlap):
        self.embeddings = HuggingFaceEmbeddings(model_name=modelName)
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.clientSettings = chromadb.config.Settings(anonymized_telemetry=False)
        self.vectorStore = None
        self.metadataPath = None
        self.persistPath = None
        self.docsStats = {}

    # can override for different loader and splitter
    def createStore(self, docsPath, persistPath=None):
        if self.vectorStore:
            raise ValueError("Already using a vector store, must delete it first")
        self.docsPath = Path(docsPath)
        self.persistPath = persistPath
        self.texts = []

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
        self.docsStats['text'] = {'docs': txtDocs, 'pages': numPages, 'chunks': len(self.texts)}
        logger.info(f"# Text Docs: {txtDocs}; # Pages: {numPages}; # Texts: {len(self.texts)}")

        pdfDocs = 0
        numPages = 0
        for filePath in self.docsPath.glob("**/*.pdf", case_sensitive=False):
            loader = PyPDFium2Loader(str(filePath))
            pages = loader.load()
            numPages += len(pages)
            self.texts += textSplitter.split_documents(pages)
            pdfDocs += 1
        self.docsStats['pdf'] = {'docs': pdfDocs, 'pages': numPages, 'chunks': len(self.texts)}
        logger.info(f"Number of pdf Docs: {pdfDocs}; # Pages: {numPages}; # Texts: {len(self.texts)}")
        if not txtDocs and not pdfDocs:
            logger.error("No documents found")
            return None

        self.vectorStore = Chroma.from_documents(self.texts, embedding=self.embeddings,
                                                 persist_directory=self.persistPath,
                                                 client_settings=self.clientSettings)

        self.metadataPath = Path(self.persistPath) / "metadata"
        metadata = {'docsPath': self.docsPath.as_posix(), 'chunkSize': self.chunkSize,
                    'chunkOverlap': self.chunkOverlap, 'docsStats': self.docsStats}
        logger.info(f"Emeddings Store metadata: {metadata}")
        if self.metadataPath.exists():
            logger.warning(f"Metadata file already exists, will be overwritten: {self.metadataPath}")
        with open(self.metadataPath, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
        logger.debug(f"Create Embeddings Store metadata file: {self.metadataPath}")
        return metadata

    def useStore(self, persistPath):
        if self.vectorStore:
            logger.error("Already using a vector store, can't create another one without first doing a delete")
            return None
        if not persistPath:
            logger.error("Must provide path to saved Emeddings Store")
            return None
        self.metadataPath = Path(persistPath) / "metadata"
        if not os.path.exists(self.metadataPath):
            logger.error(f"Invalid path to saved Emeddings Store: {self.metadataPath}")
            self.metadataPath = None
            return None
        logger.debug(f"Use Existing Embeddings Store: {self.metadataPath}")
        self.vectorStore = Chroma(embedding_function=self.embeddings,
                                  persist_directory=persistPath,
                                  client_settings=self.clientSettings)
        with open(self.metadataPath, 'r') as metadataFile:
            metadata = json.load(metadataFile)
        self.docsPath = metadata['docsPath']
        self.chunkSize = metadata['chunkSize']
        self.chunkOverlap = metadata['chunkOverlap']
        self.docsStats = metadata['docsStats']
        return metadata

    def deleteStore(self):
        if self.persistPath and os.path.exists(self.persistPath):
            logger.info(f"Deleting embeddings store: {self.persistPath}")
            shutil.rmtree(self.persistPath)
        else:
            logger.warning("No save path for Embeddings Store, nothing to delete")
        self.vectorStore = None

    #### FIXME allow for different types similarity functions -- e.g., dot and cosine
    def getContext(self, question, maxContext, threshold):
        #### TODO try using a retriever instead
#        retriever = self.vectorStore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.21, 'k': 5}')
        results = self.vectorStore.similarity_search_with_score(question, k=(maxContext // self.chunkSize))
        logger.info(f"# Docs retrieved: {len(results)}")
        chunks = [chunk for chunk, score in results]
        titles = {chunk.metadata['source']: sum(1 for item in chunks if item.metadata['source'] == chunk.metadata['source']) for chunk in chunks}
        logger.info(titles)
        scores = [(chunk.metadata['source'], score) for chunk, score in results]
        logger.info(scores)
        #### N.B. ChromaDB uses cosine distance, so lower means more similar
        filteredChunks = [chunk for chunk, score in results if score <= threshold]
        context = "\n".join([chunk.page_content for chunk in filteredChunks])
        logger.info(f"Size of thresholded (<= {threshold}) context: {len(context)} Bytes")
        return context
