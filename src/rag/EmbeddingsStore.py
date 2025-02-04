################################################################################
#
# Library that Implements a Base Class for the Document Embeddings Store
#
################################################################################

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import unstructured


class EmbeddingsStore():
    def __init__(self, K, threshold):
        self.vectorStore = None
        self.K = K
        self.threshold = threshold
        #### TODO make the embedding model be selectable
        self.embeddings = HuggingFaceEmbeddings()  # defaults to sentence-transformers/all-mpnet-base-v2

    # can override for different loader and splitter
    def createStore(self, docsPath, chunkSize, chunkOverlap, persistPath=None):
        if self.vectorStore:
            raise ValueError("Already using a vector store")

        self.docsPath = Path(docsPath)
        self.chunkSize = chunkSize if not persistPath else None
        self.chunkOverlap = chunkOverlap if not persistPath else None
        self.persistPath = persistPath
        self.texts = []

        textSplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
                                                      chunk_size=chunkSize,
                                                      chunk_overlap=chunkOverlap)
        for filePath in self.docsPath.glob("**/*.txt", case_sensitive=False):
            loader = TextLoader(str(filePath))
            pages = loader.load()
            self.texts += textSplitter.split_documents(pages)

        for filePath in self.docsPath.glob("**/*.pdf", case_sensitive=False):
            loader = PyPDFium2Loader(str(filePath))
            pages = loader.load()
            self.texts += textSplitter.split_documents(pages)

        #### TODO make vector store selectable
        self.vectorStore = Chroma.from_documents(self.texts, embedding=self.embeddings, persist_directory=self.persistPath)

    def useStore(self, persistPath):
        if self.vectorStore:
            raise ValueError("Already using a vector store")
        self.vectorStore = Chroma(embedding_function=self.embeddings, persist_directory=persistPath)

    def setK(self, K):
        self.K = K

    def getK(self):
        return(self.K)

    def setThreshold(self, threshold):
        self.threshold = threshold

    def getThreshold(self, threshold):
        return(self.threshold)

    def getContext(self, question):
        # retrieve relevant context from the knowledge base/vector store
        docs = self.vectorStore.similarity_search(question, k=self.K)
        #### TODO add thresholds
        #### N.B. ChromaDB uses cosine distance, so lower means more similar
        #### THRESH = 0.3
#        results = self.vectorStore.similarity_search_with_score(question, k=kVal)
#        filteredResults = [doc for doc, score in results if score <= THRESH]
        #### alternatively
#        retriever = self.vectorStore.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.21, 'k': 5}')
        context = "\n".join([doc.page_content for doc in docs])
        return context
