################################################################################
#
# Library that Implements a Base Class for the Document Embeddings Store
#
################################################################################

from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingsStore():
    def __init__(self):
        self.vectorStore = None
        #### TODO make the embedding model be selectable
        self.embeddings = HuggingFaceEmbeddings()  # defaults to sentence-transformers/all-mpnet-base-v2

    # can override for different loader
    def _loadDocs(self, docsPath):
        loader = TextLoader(docsPath)
        return loader.load()

    # can override for different splitter
    def _splitDocs(self, chunkSize, chunkOverlap):
        textSplitter = CharacterTextSplitter(chunk_size=chunkSize,
                                             chunk_overlap=chunkOverlap)
        return textSplitter.split_documents(self.docs)

    def createStore(self, docsPath, chunkSize, chunkOverlap, persistPath=None):
        if self.vectorStore:
            raise ValueError("Already using a vector store")
        self.docsPath = Path(docsPath)
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.persistPath = persistPath
        self.docs = self._loadDocs(self.docsPath)
        self.texts = self._splitDocs(chunkSize, chunkOverlap)
        #### TODO make vector store selectable
        self.vectorStore = Chroma.from_documents(self.texts, embedding=self.embeddings, persist_directory=self.persistPath)

    def useStore(self, persistPath):
        if self.vectorStore:
            raise ValueError("Already using a vector store")
        self.vectorStore = Chroma(embedding_function=self.embeddings, persist_directory=persistPath)

    def getContext(self, question, kVal=None, threshold=None):
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
        return content


'''
    def load(self):
        if self.file_path.suffix.lower() == '.pdf':
            self.loader = PyPDFium2Loader(str(self.file_path))
        elif self.file_path.suffix.lower() == '.txt':
            self.loader = TextLoader(str(self.file_path))
        else:
            raise ValueError("Unsupported file format. Only PDF and TXT files are supported.")
        return self.loader.load()

    def load_and_split(self, text_splitter=None):
        if not self.loader:
            self.load()
        return self.loader.load_and_split(text_splitter)
'''
