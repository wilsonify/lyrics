#!/bin/bash
parent_path=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd -P
)

cd "$parent_path"
pwd
ls

cd $parent_path/src/grapheme2phoneme
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/music-examples
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/phoneme2grapheme
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/recurrent
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/scrape-lyrics
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt
