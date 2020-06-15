FROM python:3

RUN pip install Werkzeug Flask flask-cors numpy Keras gevent pillow h5py tensorflow
WORKDIR /usr/src/app

CMD [ "python" , "app.py"]
