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
    "chunkSize": 2000,
    "confFile": ".rag.conf",
    "docPath": None,
    "embeddingModel": "all-mpnet-base-v2",
    "globalContext": "",
    "interactive": True,
    "numRetrieves": 4,
    "logLevel": DEF_LOG_LEVEL,
    "logFile": None,
    "model": "deepseek-r1:8b",
    "chuckOverlap": 0,
    "printThoughts": False,
    "query": None,
    "similarity": "Cosine",
    "vectorStore": "ChromaDB",
    "threshold": None,
    "verbose": False
}


def getOpts():
    logging.basicConfig(level=DEF_LOG_LEVEL)

    usage = f"Usage: {sys.argv[0]} [-v] [-C <chunkSize>] [-c <confFile>] \
[-d <docPath>] [-e <embeddingModel>] [-g <globalContext>] [-i] [-k <numToRetrieve>] \
[-L <logLevel>] [-l <logFile>] [-m <model>] [-o <chunkOverlap>] [-p <printThoughts>] \
[-q <query>] [-s <vectorStore>] [-t <relevanceThreshold>]"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-C", "--chunkSize", action="store", type=int,
        help="Max number of bytes in each document chunk")
    ap.add_argument(
        "-c", "--confFile", action="store", type=str,
        help="Path to file with configuration info")
    ap.add_argument(
        "-d", "--docPath", action="store", type=str,
        help="Path to directory for (.txt and .pdf) context files")
    ap.add_argument(
        "-e", "--embeddingModel", action="store", type=str,
        choices=["all-mpnet-base-v2", "????"],
        help="Name of the HuggingFace sentence-transformer embedding model to use")
    ap.add_argument(
        "-g", "--globalContext", action="store", type=str,
        help="System instructions to be applied to all queries")
    ap.add_argument(
        "-i", "--interactive", action="store_true", default=False,
        help="Take queries from, and print results to, console")
    ap.add_argument(
        "-k", "--numRetrieves", action="store", type=int,
        help="Number of document chunks to retrieve for a query (int)")
    ap.add_argument(
        "-L", "--logLevel", action="store", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level")
    ap.add_argument(
        "-l", "--logFile", action="store", type=str,
       help="Path to location of logfile (create it if it doesn't exist)")
    ap.add_argument(
        "-m", "--model", action="store", type=str,
        help="Name of Ollama LLM to use")
    ap.add_argument(
        "-o", "--chuckOverlap", action="store", type=int,
        help="Number of bytes of overlap between adjacent document chunks")
    ap.add_argument(
        "-p", "--printThoughts", action="store_true", default=False,
        help="Enable printing of chain of thought output from model")
    ap.add_argument(
        "-q", "--query", action="store", type=str,
        help="Question to ask of the model")
    ap.add_argument(
        "-S", "--similarity", action="store", type=str,
        choices=["Cosine", "DotProduct"],
        help="Name of the similarity metric to be used for embeddings")
    ap.add_argument(
        "-s", "--vectorStore", action="store", type=str,
        choices=["ChromaDB", "FAISS"],
        help="Name of the Vector Store to use to store and access document embeddings")
    ap.add_argument(
        "-t", "--threshold", action="store", type=int,
        help="Threshold above/below (depending on the similarity metric being used) which retrievals are considered valid")
    ap.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Print debug info")
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


def run(options):
    print("DONE")

if __name__ == "__main__":
    opts = getOpts()
    run(opts)
