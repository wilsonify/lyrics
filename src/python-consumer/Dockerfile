FROM python:3.7-alpine
ENV AMQP_HOST "172.17.0.1"
ENV AMQP_PORT "32777"
ADD python_consumer /python_consumer
WORKDIR /python_consumer
RUN chmod 644 python_consumer/consumer.py
RUN pip install -r requirements.txt
RUN python setup.py install
RUN python -m nltk.downloader cmudict
CMD ["python", "python_consumer/consumer.py"]