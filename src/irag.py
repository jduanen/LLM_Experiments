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


DEF_LOG_LEVEL = "WARNING"

DEFAULTS = {
    "chunkOverlap": 0,
    "chunkSize": 2000,
    "confFile": ".rag.conf",
    "docPath": None,
    "embeddingModel": "all-mpnet-base-v2",
    "globalContext": "",
    "logLevel": DEF_LOG_LEVEL,
    "logFile": None,
    "model": "deepseek-r1:8b",
    "numRetrieves": 4,
    "printThoughts": False,
    "query": None,
    "saveEmbeddingsFile": False,
    "similarity": "Cosine",
    "threshold": None,
    "useEmbeddingsFile": False,
    "vectorStore": "ChromaDB",
    "verbose": False
}


#### TODO instantiate desired vector store subclass

#### TODO subclass rag and create myRag that has this method and takes the given vector store
#    def question(self, query):
#        return self.askQuestion(query, options['numRetrieves'], options['printThoughts'])

def getOpts():
    logging.basicConfig(level=DEF_LOG_LEVEL)

    usage = f"Usage: {sys.argv[0]} [-v] [-c <confFile>] [-L <logLevel>] [-l <logFile>] [-m <model>] \
[-q <query>] [-g <globalContext>] [-p <printThoughts>] [-k <numRetrieves>] [-t <threshold>] \
[-E <saveEmbeddingsFile>] [-u <useEmbeddingsFile>] \
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
        "-E", "--saveEmbeddingsFile", action="store", type=str,
        help="Path to where embeddings store is to be saved")
    embeddingsGroup.add_argument(
        "-u", "--useEmbeddingsFile", action="store", type=str,
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

    return conf

#### TODO figure out switch constraints
#### TODO add '-E <vectorStoreFile>' to save vectorStoreName, embeddingModel, docPath, chuckOverlap, chunkSize
#### TODO add '-u <vectorStoreFile>' to use saved vectorStore
#### TODO enforce cli input constraints and test configs file for constraint violations
####  * if -E then -d, -e, -o are required and -s is not allowed
####  * if -u then -S, -d, -e, -o, -s are not allowed
####  * if -s: -d, -e not allowed
####  * -d: not allowed if -s, otherwise required
####  * -e: not allowed if -s, otherwise required

def run(options):
    if options['useEmbeddingsFile']:
        print("Use EmbeddingsFile: TBD")
    else:
        print("Create Embeddings Store: TBD")
        '''
        vectorStore = ????(options['vectorStore'], options[options['docPath']],
                           options['embeddingModel'], options['chunkOverlap'], options[''])
        '''
    if options['saveEmbeddingsFile']:
        print("Save EmbeddingsFile: TBD")

    print("Create Retriever: TBD")
    '''
    retreiver = ????(options['numToRetrieve'], options['relevanceThreshold'],
                     options['globalContext'], options[''], options[''])
    '''
    print("Create RAG: TBD")
    '''
    rag = RetrievalAugmentedGenerator(vectorStore, retreiver)
    if options['query']:
        print(f"Question: {options['query']}")
        thoughts, answer = rag.question(options['query'])
        if options['printThoughts']:
            print(f"Thoughts: {thoughts}")
        print(f"Answer: {answer}")
    else:
        while True:
            query = input("Question: ")
            if not query:
                break
            thoughts, answer = rag.question(query)
            if options['printThoughts']:
                print(f"Thoughts: {thoughts}")
            print(f"Answer: {answer}")
    '''
    logging.debug("Exiting")


if __name__ == "__main__":
    opts = getOpts()
    run(opts)
