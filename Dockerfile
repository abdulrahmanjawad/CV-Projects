FROM python:3.7.9-slim

# ENV PYTHONPATH=/usr/lib/python3.9/site-packages
EXPOSE 8000 
WORKDIR /app
# Copy requirements from host, to docker container in /app 
COPY ./requirements.txt .
# Copy everything from ./src directory to /app in the container
COPY ./app . 
RUN apt update \
    && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 
    # && apt-get install -y wget \
    # && wget --no-check-certificate https://pjreddie.com/media/files/yolov3.weights  && mv yolov3.weights /app/models/ \
    # && wget --no-check-certificate https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg && mv yolov3.cfg /app/models/ \
RUN  pip install --upgrade pip \ 
    && pip install -r requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app", "--reload"]