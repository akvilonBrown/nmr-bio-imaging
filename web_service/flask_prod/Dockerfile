FROM iaroslavplutenko/nmr-web:1.1
MAINTAINER Iaroslav Plutenko "plutenko@ucu.edu.ua"
#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential
#COPY requirements.txt /app

RUN rm -rf /app/*
COPY ./web /app
RUN rm -rf /app/files/input/*
RUN rm -rf /app/files/output/*
WORKDIR /app
#RUN pip install -r requirements.txt
#RUN pip3.6 install --user --upgrade werkzeug
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["home.py"]
