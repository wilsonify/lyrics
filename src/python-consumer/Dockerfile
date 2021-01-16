FROM python:3.9-slim-buster
ENV AMQP_HOST "localhost"
ENV AMQP_PORT "5672"
ADD . /src/python-consumer
WORKDIR /src/python-consumer
RUN pip install -r requirements.txt
RUN python setup.py install
RUN python -m nltk.downloader cmudict
ENTRYPOINT ["python"]
CMD ["-m", "python_consumer"]