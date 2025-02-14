#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# CLI for creating embeddings and exploring similarity-based searches in vector stores
#
################################################################################


import argparse
import json
import logging
import os
import sys
import time
import yaml

import pdb  ## pdb.set_trace()  #### TMP TMP TMP

from rag.EmbeddingsStore import EmbeddingsStore


DEF_LOG_LEVEL = "WARNING"

DEF_CHUNK_SIZE = 1000    # FIXME random choice -- come up with better value
DEF_CHUNK_OVERLAP = 500  # FIXME random choice -- come up with better value

DEF_MAX_CONTEXT = 16000  # FIXME random choice -- come up with better value
DEF_THRESHOLD = 1.15     # N.B. Assumes ChromaDB uses Cosine similarity

DEF_MODEL = "all-mpnet-base-v2"

DEFAULTS = {
    "chunkOverlap": DEF_CHUNK_OVERLAP,
    "chunkSize": DEF_CHUNK_SIZE,
    "confFile": ".embedx.conf",
    "docsPath": None,
    "forceSaveEmbeddings": False,
    "logLevel": DEF_LOG_LEVEL,
    "logFile": None,
    "maxContext": DEF_MAX_CONTEXT,
    "model": DEF_MODEL,
    "query": None,
    "saveEmbeddingsPath": None,
    "similarity": "Cosine",
    "threshold": DEF_THRESHOLD,
    "useEmbeddingsPath": None,
    "vectorStore": "ChromaDB",
    "verbose": False
}


def getOpts():
    usage = f"Usage: {sys.argv[0]} [-v] [-c <confFile>] [-L <logLevel>] [-l <logFile>] [-m <model>] \
[-q <query>] [-x <maxContext>] [-t <threshold>] \
[-E <saveEmbeddingsPath>] [-f] [-u <useEmbeddingsPath>] [-s <vectorStore>] [-S <similarity>] \
[-d <docsPath>] [-C <chunkSize>] [-O <chunkOverlap>]"

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
        "-m", "--model", action="store", type=str,
        choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2"],
        help="Name of the HuggingFace sentence-transformer embedding model to use")
    generalGroup.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Print debug info")

    queryGroup = ap.add_argument_group("Query Options")
    queryGroup.add_argument(
        "-q", "--query", action="store", type=str,
        help="Question to ask of the model")
    queryGroup.add_argument(
        "-x", "--maxContext", action="store", type=int,
        help="Maximum number of bytes in the retrieved context")
    queryGroup.add_argument(
        "-t", "--threshold", action="store", type=float,
        help="Threshold above/below (depending on the similarity metric being used) which retrievals are considered valid")

    vectorStoreGroup = ap.add_argument_group("Vector Store Options")
    vectorStoreGroup.add_argument(
        "-E", "--saveEmbeddingsPath", action="store", type=str,
        help="Path to where embeddings store is to be saved")
    vectorStoreGroup.add_argument(
        "-f", "--forceSaveEmbeddings", action="store_true",
        help="Overwrite existing embeddings store")
    vectorStoreGroup.add_argument(
        "-u", "--useEmbeddingsPath", action="store", type=str,
        help="Path to where embeddings store is to be obtained")
    vectorStoreGroup.add_argument(
        "-S", "--similarity", action="store", type=str,
        choices=["Cosine", "DotProduct"],
        help="Name of the similarity metric to be used for embeddings")
    vectorStoreGroup.add_argument(
        "-s", "--vectorStore", action="store", type=str,
        choices=["ChromaDB", "FAISS"],
        help="Name of the Vector Store to use to store and access document embeddings")

    embeddingsGroup = ap.add_argument_group("Embeddings Options")
    embeddingsGroup.add_argument(
        "-d", "--docsPath", action="store", type=str,
        help="Path to directory for (.txt and .pdf) context files")
    embeddingsGroup.add_argument(
        "-C", "--chunkSize", action="store", type=int,
        help="Max number of bytes in each document chunk")
    embeddingsGroup.add_argument(
        "-O", "--chuckOverlap", action="store", type=int,
        help="Number of bytes of overlap between adjacent document chunks")

    cliOpts = ap.parse_args().__dict__

    conf = {'cli': cliOpts, 'confFile': {}, 'config': {}}
    if cliOpts['confFile']:
        if not os.path.exists(cliOpts['confFile']):
            print(f"ERROR: Invalid configuration file: {cliOpts['confFile']}")
            exit(1)
        with open(cliOpts['confFile'], "r") as confsFile:
            confs = list(yaml.load_all(confsFile, Loader=yaml.Loader))
            if len(confs) >= 1:
                conf['confFile'] = confs[0]
                if len(confs) > 1:
                    print(f"WARNING: Multiple config docs in file. Using the first one")

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

    if cliOpts['verbose'] > 3:
        json.dump(conf, sys.stdout, indent=4, sort_keys=True)
        print("")

    if conf['logFile'] and (conf['logFile'] != '-'):
        logging.basicConfig(filename=conf['logFile'], level=conf['logLevel'])
    else:
        logging.basicConfig(level=conf['logLevel'])
    logging.debug(f"Logging to {conf['logFile']} at level {conf['logLevel']}")

    # check for consistency among switches
    if conf['useEmbeddingsPath']:
        for k in ('docsPath', 'model', 'chunkOverlap', 'chunkSize'):
            if conf['cli'].get(k):
                logging.warning(f"Using saved embeddings, ignoring {k}")

    if conf['forceSaveEmbeddings'] and not conf['saveEmbeddingsPath']:
        logging.warning("No save path given, ignoring overwrite flag")
    if conf['saveEmbeddingsPath'] and os.path.exists(conf['saveEmbeddingsPath']) and not conf['forceSaveEmbeddings']:
        logging.error(f"Embeddings Store already exists ({conf['saveEmbeddingsPath']})")
        exit(1)
    return conf

def run(options):
    def _queryInfo(query, maxContext, threshold):
        print("Query Info:")
        print(f"    Query: {query}")
        print("TBD")

    def _contextInfo(context):
        print("Context Info:")
        print("TBD")

    startTime = time.time()
    embeddingsStore = EmbeddingsStore(options['model'], options['chunkSize'],
                                      options['chunkOverlap'])
    if options['useEmbeddingsPath']:
        metadata = embeddingsStore.useStore(options['useEmbeddingsPath'])
        if not metadata:
            logging.error(f"Failed to get saved Embeddings Store ({options['useEmbeddingsPath']})")
            exit(1)
    else:
        # N.B. if saveEmbeddingsPath is None, then don't persist the vector store
        if options['forceSaveEmbeddings'] and options['saveEmbeddingsPath']:
            embeddingsStore.deleteStore()
        metadata = embeddingsStore.createStore(options['docsPath'], options['saveEmbeddingsPath'])
    embStoreSetupTime = time.time() - startTime
    logging.info(f"EmbeddingsStore setup time: {embStoreSetupTime:.4f} secs")

    if options['verbose']:
        print(f"    Embeddings Model:      {options['model']}")
        if options['useEmbeddingsPath']:
            print(f"    Embeddings Store:      {options['useEmbeddingsPath']}")
        else:
            if options['saveEmbeddingsPath']:
                print(f"    Save Embeddings Store: {options['saveEmbeddingsPath']}")
            else:
                print("    Not saving Embeddings Store")
        print(f"    Docs Path:             {metadata['docsPath']}")
        print(f"    Doc Chunk Size:        {metadata['chunkSize']}")
        print(f"    Chunk Overlap:         {metadata['chunkOverlap']}")
        print("Doc Stats:")
        print(f"   {json.dumps(metadata['docsStats'], indent=4, sort_keys=True).replace('\n', '\n    ')}")
        print("Timing:")
        print(f"    EmbeddingsStore setup: {embStoreSetupTime:.4f} secs")

    query = options['query']
    if query:
        startTime = time.time()
        context = embeddingsStore.getContext(query, options['maxContext'], options['threshold'])
        getContextTime = time.time() - startTime
        if options['verbose']:
            print(f"    Get Context:           {getContextTime:.4f} secs")
            _queryInfo(query, options['maxContext'], options['threshold'])
        _contextInfo(context)
    else:
        while True:
            query = input("Question: ")
            if not query:
                break
            startTime = time.time()
            context = embeddingsStore.getContext(query, options['maxContext'], options['threshold'])
            getContextTime = time.time() - startTime
            if options['verbose']:
                print(f"    Get Context:           {getContextTime:.4f} secs")
                _queryInfo(query, options['maxContext'], options['threshold'])
            _contextInfo(context)
    logging.debug("Exiting")


if __name__ == "__main__":
    opts = getOpts()
    run(opts)

