FROM python:3.7-alpine
ADD . /grapheme2phoneme
WORKDIR /grapheme2phoneme
RUN chmod 644 grapheme2phoneme/consumer.py
RUN pip install -r requirements.txt
RUN python setup.py install
RUN python -m nltk.downloader cmudict
CMD ["python", "grapheme2phoneme/consumer.py"]