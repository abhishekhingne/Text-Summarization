FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN apt-get install libffi6 libffi-dev -y
RUN apt-get install libssl-dev -y
COPY src /app
WORKDIR /app/src
RUN pip3 install -r requirements.txt
CMD ["python3","app.py"]