FROM python:3.6
MAINTAINER Iaroslav Plutenko "plutenko@ucu.edu.ua"
#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential
COPY ./web /app
COPY requirements.txt /app
WORKDIR /app
RUN pip install -r requirements.txt
#RUN pip3.6 install --user --upgrade werkzeug
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["home.py"]
