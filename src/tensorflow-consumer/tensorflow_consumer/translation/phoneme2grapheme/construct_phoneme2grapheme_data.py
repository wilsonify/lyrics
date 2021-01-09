# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import glob
import re

from python_consumer import consumer
from tensorflow_consumer.config import DATA_DIR

graphemefiles = [
    txtfilepath for txtfilepath in glob.glob(f"{DATA_DIR}/beatles_lyrics/*.txt")
]
phonemefiles = [
    txtfilepath for txtfilepath in glob.glob(f"{DATA_DIR}/beatles_lyrics_phoneme/*.txt")
]

total = len(graphemefiles)
count = 0
for graphemefile in graphemefiles:
    count += 1
    name_match = re.search(
        string=graphemefile, pattern=f"{DATA_DIR}/beatles_lyrics/(.*)\.txt"
    )
    name_str = name_match.group(1)
    combined_file = f"{DATA_DIR}/beatles_lyrics_combined/{name_str}_combined.txt"
    if count % 10 == 0:
        print(f"graphemefile = {graphemefile}")
        print(f"combined_file = {combined_file}")
        print(f"{count}/{total} = {count / total:0.2f}")
    with open(graphemefile, "r") as graphemefile_open:
        with open(combined_file, "w") as combined_file_open:
            for line in graphemefile_open.readlines():
                line = line.replace("\n", "")
                line = line.replace("Title:", "Title: ")
                if not line:
                    continue
                phonemes_list = consumer.graphemes2phonemes(line)
                phonemes_str = consumer.reduce_to_string(phonemes_list)
                line_to_write = f"{line} \t {phonemes_str}"
                combined_file_open.write(line_to_write)
                combined_file_open.write("\n")
