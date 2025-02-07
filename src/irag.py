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
import re
import sys
import time
import yaml

import pdb  ## pdb.set_trace()  #### TMP TMP TMP

from rag.RetrievalAugmentedGeneration import RetrievalAugmentedGeneration
from rag.EmbeddingsStore import EmbeddingsStore


DEF_LOG_LEVEL = "WARNING"

DEF_LLM_NAME = "deepseek-r1:8b"

DEF_CHUNK_SIZE = 1000
DEF_CHUNK_OVERLAP = 500

DEF_EMBD_MODEL = "all-mpnet-base-v2"

DEF_OUTPUT_FORMAT = "HUMAN"

DEF_TEMPERATURE = 0.4
DEF_NUM_SENTENCES = 10
DEF_PERSONA = "physicist"  # "programmer"

# instruct model to respond based only on the retrieved context
DEF_GLOBAL_CONTEXT = f"""
You are an experienced ${DEF_PERSONA}, speaking to another experienced ${DEF_PERSONA}.
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under {DEF_NUM_SENTENCES} sentences.
[temperature: {DEF_TEMPERATURE}]
"""

DEFAULTS = {
    "chunkOverlap": DEF_CHUNK_OVERLAP,
    "chunkSize": DEF_CHUNK_SIZE,
    "confFile": ".rag.conf",
    "docPath": None,
    "embeddingModel": DEF_EMBD_MODEL,
    "globalContext": DEF_GLOBAL_CONTEXT,
    "logLevel": DEF_LOG_LEVEL,
    "logFile": None,
    "model": DEF_LLM_NAME,
    "numRetrieves": 4,
    "outputFormat": DEF_OUTPUT_FORMAT,
    "printThoughts": False,
    "query": None,
    "saveEmbeddingsPath": None,
    "similarity": "Cosine",
    "threshold": None,
    "useEmbeddingsPath": None,
    "vectorStore": "ChromaDB",
    "verbose": False
}


def getOpts():
    usage = f"Usage: {sys.argv[0]} [-v] [-c <confFile>] [-L <logLevel>] [-l <logFile>] [-m <model>] \
[-q <query>] [-g <globalContext>] [-p <printThoughts>] [-k <numRetrieves>] [-t <threshold>] \
[-E <saveEmbeddingsPath>] [-u <useEmbeddingsPath>] \
[-d <docPath>] [-e <embeddingModel>] [-s <vectorStore>] [-S <similarity>] [-C <chunkSize>] [-O <chunkOverlap>] \
[-o <outputFormat>]"

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
        choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2"],
        help="Name of the HuggingFace sentence-transformer embedding model to use")
    embeddingsGroup.add_argument(
        "-C", "--chunkSize", action="store", type=int,
        help="Max number of bytes in each document chunk")
    embeddingsGroup.add_argument(
        "-O", "--chuckOverlap", action="store", type=int,
        help="Number of bytes of overlap between adjacent document chunks")
    embeddingsGroup.add_argument(
        "-S", "--similarity", action="store", type=str,
        choices=["Cosine", "DotProduct"],
        help="Name of the similarity metric to be used for embeddings")
    embeddingsGroup.add_argument(
        "-s", "--vectorStore", action="store", type=str,
        choices=["ChromaDB", "FAISS"],
        help="Name of the Vector Store to use to store and access document embeddings")
    outputGroup = ap.add_argument_group("Output Options")
    outputGroup.add_argument(
        "-o", "--outputFormat", action="store", type=str,
        choices=["HUMAN", "JSON"],
        help="Format of output")
    outputGroup.add_argument(
        "-p", "--printThoughts", action="store_true", default=False,
        help="Enable printing of chain of thought output from model")
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
        for k in ('docPath', 'embeddingModel', 'chunkOverlap', 'chunkSize'):
            if not conf.get(k):
                logging.warning(f"Using saved embeddings, ignoring {k}")
    return conf

def run(options):
    def handleResponse():
        #### FIXME use match and get the two parts without the delimiters
        pattern = f"({re.escape('<think>')}.*?{re.escape('</think>')})"
        parts = re.split(pattern, response['response'], maxsplit=1, flags=re.DOTALL)
        if len(parts) == 3:
            thoughts = parts[1]
            answer = parts[2]
        elif len(parts) == 1:
            thoughts = None
            answer = parts[0]
        else:
            logging.error(f"Confused response: {parts}")
            thoughts = None
            answer = None
        stats = {
            'totalDuration': response['total_duration'],
            'loadDuration': response['load_duration'],
            'promptEvalCount': int(response['prompt_eval_count']),
            'promptEvalDuration': response['prompt_eval_duration'],
            'promptEvalRate': (response['prompt_eval_count'] * 1000000000) / response['prompt_eval_duration'],
            'evalCount': int(response['eval_count']),
            'evalDuration': response['eval_duration'],
            'evalRate': (response['eval_count'] * 1000000000.0) / response['eval_duration'],
        }

        if options['outputFormat'] == "HUMAN":
            print(f"\nQuestion: {query}")
            if options['verbose'] > 1:
                print(f"\nContext:\n{metadata['context']}")
            if options['printThoughts']:
                print(f"\nThoughts:\n{thoughts}")
            print(f"\nAnswer: {answer}")
            if options['verbose'] > 1:
                print("\nPerformance:")
                print(f"    Setup Embs:  {metadata['embStoreSetupTime']:.2f} secs")
                print(f"    Get Context: {metadata['getContextTime']:.2f} secs")
                print(f"    Generate:    {metadata['generateTime']:.2f} secs")
                print("\nLLM Generate Stats:")
                print(f"    Total Duration:       {(stats['totalDuration'] / 1000000000.0):.2f} secs")
                print(f"    Load Duration:        {(stats['loadDuration'] / 1000000.0):.2f} msecs")
                print(f"    Prompt Eval Tokens:   {int(stats['promptEvalCount'])} tokens")
                print(f"    Prompt Eval Duration: {(stats['promptEvalDuration'] / 1000000.0):.2f} msecs")
                print(f"    Prompt Eval Rate:     {stats['promptEvalRate']:.2f} tokens/sec")
                print(f"    Eval Count:           {int(stats['evalCount'])} tokens")
                print(f"    Eval Duration:        {(stats['evalDuration'] / 1000000000.0):.2f} secs")
                print(f"    Eval Rate:            {stats['evalRate']:.2f} tokens/sec")
                totalTime = metadata['embStoreSetupTime'] + metadata['getContextTime'] + metadata['generateTime']
                print(f"\nTotal time: {totalTime:.2f} secs")
            print("\n")
        elif options['outputFormat'] == "JSON":
            outDict = {'query': query, 'context': metadata['context'], 'answer': answer, 'reason': response.done_reason}
            if options['printThoughts']:
                outDict['thoughts'] = thoughts
            if options['verbose'] > 1:
                outDict['stats'] = stats
            outDict['embStats'] = {'embStoreSetupTime': metadata['embStoreSetupTime'],
                                   'getContextTime': metadata['getContextTime'],
                                   'generateTime': metadata['generateTime']}
            print(json.dumps(outDict, indent=4, sort_keys=True))
            print("\n")
        else:
            logging.warning(f"Unknown output format: {options['outputFormat']}")

    startTime = time.time()
    embeddingsStore = EmbeddingsStore(options['embeddingModel'], options['numRetrieves'], options['threshold'])
    if options['useEmbeddingsPath']:
        embeddingsStore.useStore(options['useEmbeddingsPath'])
    else:
        # N.B. if saveEmbeddingsPath is None, then don't persist the vector store
        embeddingsStore.createStore(options['docPath'], options['chunkSize'],
                                    options['chunkOverlap'], options['saveEmbeddingsPath'])
    rag = RetrievalAugmentedGeneration(embeddingsStore, options['model'], options['globalContext'])
    embStoreSetupTime = time.time() - startTime

    query = options['query']
    if query:
        response, metadata = rag.answerQuestion(query)
        metadata['embStoreSetupTime'] = embStoreSetupTime
        handleResponse()
    else:
        while True:
            query = input("Question: ")
            if not query:
                break
            response, metadata = rag.answerQuestion(query)
            metadata['embStoreSetupTime'] = embStoreSetupTime
            handleResponse()
    logging.debug("Exiting")


if __name__ == "__main__":
    opts = getOpts()
    run(opts)
