FROM python:3.7-alpine
ADD . /code
WORKDIR /code
RUN chmod 644 /code/app.py
RUN pip install pika nltk
RUN python -m nltk.downloader all
CMD ["python", "consumer.py"]
EXPOSE 5000
