FROM python:3.7

COPY requirements.txt requirements.txt

WORKDIR /app

RUN pip install cmake==3.18.2.post1 dlib==19.21.0 face-recognition==1.3.0 face-recognition-models==0.3.0 numpy==1.18.5

CMD [ "python", "./face_recognition_practice.py" ]