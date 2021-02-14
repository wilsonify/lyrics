#!/bin/bash
parent_path=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd -P
)

cd "$parent_path"
pwd
ls

cd $parent_path/src/lyrics-api
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/python-consumer
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt

cd $parent_path/src/tensorflow-consumer
pwd
ls
python setup.py develop
python -m pip install -r requirements.txt
