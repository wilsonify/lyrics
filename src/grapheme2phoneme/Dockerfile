FROM python:3.7-alpine
ENV AMQP_HOST "172.17.0.1"
ENV AMQP_PORT "32777"
ADD . /grapheme2phoneme
WORKDIR /grapheme2phoneme
RUN chmod 644 grapheme2phoneme/consumer.py
RUN pip install -r requirements.txt
RUN python setup.py install
RUN python -m nltk.downloader cmudict
CMD ["python", "grapheme2phoneme/consumer.py"]