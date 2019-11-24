#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import os
import time
import MeCab
import wget
from multiprocessing import cpu_count
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Word2vec approach for Japanese language using Gensim.')

parser.add_argument(
    '--vectorsize',
    type=int,
    required=False,
    help='Gensim Vector Size'
)

args = parser.parse_args()

VECTORS_SIZE = args.vectorsize
INPUT_FILENAME = "jawiki-latest-pages-articles.xml.bz2"

JA_WIKI_TEXT_FILENAME = 'jawiki-latest-text.txt'
JA_WIKI_SENTENCES_FILENAME = 'jawiki-latest-text-sentences.txt'

JA_WIKI_TEXT_TOKENS_FILENAME = 'jawiki-latest-text-tokens.txt'
JA_WIKI_SENTENCES_TOKENS_FILENAME = 'jawiki-latest-text-sentences-tokens.txt'

JA_VECTORS_MODEL_FILENAME = f'ja-gensim.{VECTORS_SIZE}d.data.model'
JA_VECTORS_TEXT_FILENAME = f'ja-gensim.{VECTORS_SIZE}d.data.txt'
JA_WIKI_LATEST_URL = 'https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2'


def generate_vectors(input_filename, output_filename, output_filename_2):

    if os.path.isfile(output_filename):
        logging.info(f'Skipping generate_vectors(). File already exists: {output_filename}')
        return

    start = time.time()

    model = Word2Vec(LineSentence(input_filename),
                     size=VECTORS_SIZE,
                     window=5,
                     min_count=5,
                     workers=4,
                     iter=5)

    model.save(output_filename)
    model.wv.save_word2vec_format(output_filename_2, binary=False)

    logging.info('Finished generate_vectors(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))


def get_words(text):
    mt = MeCab.Tagger('-Owakati')

    mt.parse('')

    parsed = mt.parseToNode(text)
    components = []

    while parsed:
        components.append(parsed.surface)
        parsed = parsed.next

    return components


def tokenize_text(input_filename, output_filename):

    if os.path.isfile(output_filename):
        logging.info(f'Skipping tokenize_text(). File already exists: {output_filename}')
        return

    start = time.time()

    with open(output_filename, 'w') as out:
        with open(input_filename, 'r') as inp:
            for i, text in enumerate(inp.readlines()):
                tokenized_text = ' '.join(get_words(text))
                out.write(tokenized_text)

                if i % 100 == 0 and i != 0:
                    logging.info('Tokenized {} articles.'.format(i))
    logging.info('Finished tokenize_text(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))


def process_wiki_to_text(input_filename, output_text_filename, output_sentences_filename):

    if os.path.isfile(output_text_filename) and os.path.isfile(output_sentences_filename):
        logging.info(f'Skipping process_wiki_to_text(). Files already exist: {output_text_filename} {output_sentences_filename}')
        return

    start = time.time()
    intermediary_time = None
    sentences_count = 0

    with open(output_text_filename, 'w') as out:
        with open(output_sentences_filename, 'w') as out_sentences:
            wiki = WikiCorpus(input_filename, lemmatize=False, dictionary={}, processes=cpu_count())
            wiki.metadata = True
            texts = wiki.get_texts()

            for i, article in enumerate(texts):
                # article[1] refers to the name of the article.
                text_list = article[0]  
                sentences = text_list
                sentences_count += len(sentences)

                # Write sentences per line
                for sentence in sentences:
                    out_sentences.write((sentence + '\n'))

                # Write each page in one line
                text = ' '.join(sentences) + '\n'
                out.write(text)

                # This is just for the logging
                if i % (100 - 1) == 0 and i != 0:
                    if intermediary_time is None:
                        intermediary_time = time.time()
                        elapsed = intermediary_time - start
                    else:
                        new_time = time.time()
                        elapsed = new_time - intermediary_time
                        intermediary_time = new_time
                    sentences_per_sec = int(len(sentences) / elapsed)
                    logging.info('Saved {0} articles containing {1} sentences ({2} sentences/sec).'.format(i + 1,
                                                                                                           sentences_count,
                                                                                                           sentences_per_sec))
        logging.info('Finished process_wiki_to_text(). It took {0:.2f} s to execute.'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    if not os.path.isfile(INPUT_FILENAME):
        wget.download(JA_WIKI_LATEST_URL)

    process_wiki_to_text(INPUT_FILENAME, JA_WIKI_TEXT_FILENAME, JA_WIKI_SENTENCES_FILENAME)
    tokenize_text(JA_WIKI_TEXT_FILENAME, JA_WIKI_TEXT_TOKENS_FILENAME)

    generate_vectors(JA_WIKI_SENTENCES_FILENAME, JA_VECTORS_MODEL_FILENAME, JA_VECTORS_TEXT_FILENAME)
