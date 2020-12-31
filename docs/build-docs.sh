#!/usr/bin/env bash

python -m mkdocs build -f docs/mkdocs.yml -d docs/site

cp ./docs/index.md ./README.md
