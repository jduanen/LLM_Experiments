#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# CLI for making queries on a RAG model
#
################################################################################


import argparse
import json
import logging
import os
import sys
import yaml

import pdb  ## pdb.set_trace()  #### TMP TMP TMP

from rag.RetrievalAugmentedGeneration import RetrievalAugmentedGeneration
from rag.EmbeddingsStore import EmbeddingsStore


DEF_LOG_LEVEL = "WARNING"

DEF_LLM_NAME = "deepseek-r1:1.5b"

DEF_CHUNK_SIZE = 2000
DEF_CHUNK_OVERLAP = 0

# instruct model to respond based only on the retrieved context
DEF_GLOBAL_CONTEXT = """
You are an experienced programmer, speaking to another experienced programmer.
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under 4 sentences.
"""

DEFAULTS = {
    "chunkOverlap": DEF_CHUNK_OVERLAP,
    "chunkSize": DEF_CHUNK_SIZE,
    "confFile": ".rag.conf",
    "docPath": None,
    "embeddingModel": "all-mpnet-base-v2",
    "globalContext": DEF_GLOBAL_CONTEXT,
    "logLevel": DEF_LOG_LEVEL,
    "logFile": None,
    "model": DEF_LLM_NAME,
    "numRetrieves": 4,
    "printThoughts": False,
    "query": None,
    "saveEmbeddingsPath": False,
    "similarity": "Cosine",
    "threshold": None,
    "useEmbeddingsPath": False,
    "vectorStore": "ChromaDB",
    "verbose": False
}


def getOpts():
    logging.basicConfig(level=DEF_LOG_LEVEL)

    usage = f"Usage: {sys.argv[0]} [-v] [-c <confFile>] [-L <logLevel>] [-l <logFile>] [-m <model>] \
[-q <query>] [-g <globalContext>] [-p <printThoughts>] [-k <numRetrieves>] [-t <threshold>] \
[-E <saveEmbeddingsPath>] [-u <useEmbeddingsPath>] \
[-d <docPath>] [-e <embeddingModel>] [-s <vectorStore>] [-S <similarity>] [-C <chunkSize>] [-o <chunkOverlap>]"

    ap = argparse.ArgumentParser()
    generalGroup = ap.add_argument_group("General Options")
    generalGroup.add_argument(
        "-c", "--confFile", action="store", type=str,
        help="Path to file with configuration info")
    generalGroup.add_argument(
        "-L", "--logLevel", action="store", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level")
    generalGroup.add_argument(
        "-l", "--logFile", action="store", type=str,
       help="Path to location of logfile (create it if it doesn't exist)")
    generalGroup.add_argument(
        "-m", "--model", action="store", type=str,
        help="Name of Ollama LLM to use")
    generalGroup.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Print debug info")

    queryGroup = ap.add_argument_group("Query Options")
    queryGroup.add_argument(
        "-g", "--globalContext", action="store", type=str,
        help="System instructions to be applied to all queries")
    queryGroup.add_argument(
        "-p", "--printThoughts", action="store_true", default=False,
        help="Enable printing of chain of thought output from model")
    queryGroup.add_argument(
        "-q", "--query", action="store", type=str,
        help="Question to ask of the model")
    queryGroup.add_argument(
        "-k", "--numRetrieves", action="store", type=int,
        help="Number of document chunks to retrieve for a query (int)")
    queryGroup.add_argument(
        "-t", "--threshold", action="store", type=int,
        help="Threshold above/below (depending on the similarity metric being used) which retrievals are considered valid")

    embeddingsGroup = ap.add_argument_group("Embeddings Store Options")
    embeddingsGroup.add_argument(
        "-E", "--saveEmbeddingsPath", action="store", type=str,
        help="Path to where embeddings store is to be saved")
    embeddingsGroup.add_argument(
        "-u", "--useEmbeddingsPath", action="store", type=str,
        help="Path to where embeddings store is to be obtained")
    embeddingsGroup.add_argument(
        "-d", "--docPath", action="store", type=str,
        help="Path to directory for (.txt and .pdf) context files")
    embeddingsGroup.add_argument(
        "-e", "--embeddingModel", action="store", type=str,
        choices=["all-mpnet-base-v2", "????"],
        help="Name of the HuggingFace sentence-transformer embedding model to use")
    embeddingsGroup.add_argument(
        "-C", "--chunkSize", action="store", type=int,
        help="Max number of bytes in each document chunk")
    embeddingsGroup.add_argument(
        "-o", "--chuckOverlap", action="store", type=int,
        help="Number of bytes of overlap between adjacent document chunks")
    embeddingsGroup.add_argument(
        "-S", "--similarity", action="store", type=str,
        choices=["Cosine", "DotProduct"],
        help="Name of the similarity metric to be used for embeddings")
    embeddingsGroup.add_argument(
        "-s", "--vectorStore", action="store", type=str,
        choices=["ChromaDB", "FAISS"],
        help="Name of the Vector Store to use to store and access document embeddings")
    cliOpts = ap.parse_args().__dict__

    conf = {'cli': cliOpts, 'confFile': {}, 'config': {}}
    if cliOpts['confFile']:
        if not os.path.exists(cliOpts['confFile']):
            logging.error(f"Invalid configuration file: {cliOpts['confFile']}")
            exit(1)
        with open(cliOpts['confFile'], "r") as confsFile:
            confs = list(yaml.load_all(confsFile, Loader=yaml.Loader))
            if len(confs) >= 1:
                conf['confFile'] = confs[0]
                if len(confs) > 1:
                    logging.warning(f"Multiple config docs in file. Using the first one")

    # options precedence order: cmd line -> conf file -> defaults
    #   cliOpts: cmd line options
    #   conf: conf file options
    #   DEFAULT: default options
    def _configSelect(opt):
        if (opt in conf['cli']) and (conf['cli'][opt] is not None):
            conf[opt] = conf['cli'][opt]
        elif (opt in conf['confFile']) and (conf['confFile'][opt] is not None):
            conf[opt] = conf['confFile'][opt]
        else:
            conf[opt] = DEFAULTS[opt]

    for opt in DEFAULTS.keys():
        _configSelect(opt)

    if cliOpts['verbose'] > 2:
        json.dump(conf, sys.stdout, indent=4, sort_keys=True)
        print("")

    if conf['logFile']:
        logging.basicConfig(filename=conf['logFile'], level=conf['logLevel'])
    else:
        logging.basicConfig(level=conf['logLevel'])

    # check for consistency among switches
    if conf['useEmbeddingsPath']:
        for k in ('docPath', 'embeddingModel', 'chunkOverlap', 'chunkSize'):
            if not conf.get(k):
                logging.warning(f"Using saved embeddings, ignoring {k}")

    return conf


def run(options):
    embeddingsStore = EmbeddingsStore(options['numRetrieves'], options['threshold'])
    if options['useEmbeddingsPath']:
        print("Use saved embeddingsStore")
        embeddingStore.useStore(options['useEmbeddingsPath'])
    else:
        print(f"Create Embeddings Store: {options['docPath']}, {options['chunkSize']}, {options['chunkOverlap']}, {options['saveEmbeddingsPath']}")
        embeddingsStore.createStore(options['docPath'], options['chunkSize'],
                                    options['chunkOverlap'], options['saveEmbeddingsPath'])

    rag = RetrievalAugmentedGeneration(embeddingsStore, options['model'], options['globalContext'])
    if options['query']:
        print(f"Question: {options['query']}")
        thoughts, answer = rag.answerQuestion(options['query'])
        if options['printThoughts']:
            print(f"Thoughts: {thoughts}")
        print(f"Answer: {answer}")
    else:
        while True:
            query = input("Question: ")
            if not query:
                break
            thoughts, answer = rag.answerQuestion(query)
            if options['printThoughts']:
                print(f"Thoughts: {thoughts}")
            print(f"Answer: {answer}")
    logging.debug("Exiting")


if __name__ == "__main__":
    opts = getOpts()
    run(opts)
