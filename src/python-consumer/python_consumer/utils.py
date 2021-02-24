import logging
import os
import re
import nltk
import pika
from python_consumer import config
from python_consumer import throat

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    import nltk

    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()


def reduce_to_string(list_of_lists):
    def flatten(nested):
        if not nested:
            return nested
        if isinstance(nested[0], list):
            return flatten(nested[0]) + flatten(nested[1:])
        return nested[:1] + flatten(nested[1:])

    if isinstance(list_of_lists, str):
        return list_of_lists
    flat_list = flatten(list_of_lists)
    return " ".join(flat_list)


def word2phoneme(grapheme):
    grapheme = grapheme.lower()
    grapheme = re.sub(pattern=r'\W+', repl="", string=grapheme)
    try:
        phoneme = arpabet[grapheme][0]
    except (KeyError, IndexError):
        logging.debug("grapheme not in cmudict, try text to sound rules")
        phoneme = throat.text_to_phonemes(text=grapheme)
        phoneme = re.sub(pattern=r'\W+', repl="", string=phoneme)
        phoneme = re.sub(pattern=r'-+', repl=" ", string=phoneme)
        phoneme = re.sub(pattern=r' +', repl=" ", string=phoneme)
    logging.debug("grapheme = {}".format(grapheme))
    logging.debug("phoneme = {}".format(phoneme))
    return phoneme


def graphemes2phonemes(body):
    result = []
    if isinstance(body, str):
        body = body.split(" ")
    for word in body:
        phoneme = word2phoneme(word)
        result.append(phoneme)
    return result








def grapheme2phoneme_str(grapheme):
    phonemes_list = graphemes2phonemes(grapheme)
    phoneme = reduce_to_string(phonemes_list)
    return phoneme


def grapheme2phoneme_file(payload):
    logging.info("grapheme2phoneme")
    payload_head, payload_tail = os.path.splitext(payload)
    payload_head_head, payload_head_tail = os.path.split(payload_head)
    result_file_name = payload_head_tail + '_phoneme.txt'
    result_dir = os.path.join(config.local_data, payload_head_head + "_phoneme")
    result_path = os.path.join(result_dir, result_file_name)
    os.makedirs(result_dir, exist_ok=True)
    result = ""
    try:
        with open(payload, 'r') as requested_file:
            with open(result_path, 'a') as result_file:
                for line in requested_file:
                    logging.debug("line = {}".format(line))
                    if line.startswith("Title:"):
                        continue
                    elif re.search(string=line, pattern=r"\[.+\]"):
                        result_file.write(line + "\n")
                        continue
                    else:
                        grapheme = line
                        phonemes = graphemes2phonemes(grapheme)
                        logging.debug("grapheme = {}".format(grapheme))
                        logging.debug("phonemes = {}".format(phonemes))
                        result_file.write(reduce_to_string(phonemes) + "\n")
    except FileNotFoundError:
        logging.exception("requested file does not exist")

    return result





