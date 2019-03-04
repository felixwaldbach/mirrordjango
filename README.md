# Mirrordjango

## What does this Repository contain?
This repository holds code for a basic [Django Python Webserver}(https://www.djangoproject.com/) using a REST API. 
The api serves with routes to trigger face recognition algorithms. For details about these algorithms see [Adrian Rosebrock's 
PyImageSearch](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) blog.
Most of the code used in this approach was inspired by his blog and is working on the same principles. Thank you, Adrian!

This project was created to perform face recognition in order to authenticate user's in a [selfmade smart mirror application](https://github.com/felixwaldbach/mirrorserver) we are
currently building for university. However, since the smart mirror application itself is developed for and running on a RaspberryPi with limited processing power, the face recognition
is performed externally on this little server. This project has only been tested on a 2018 MacBook Pro so far.

## Get started
First of all, ensure to have a running python instance ready on your machine. We used an anaconda virtual environment with Python 3.6 installed.
To execute the project, install following packages using pip:
```pip install django
pip install django-cors-headers
pip install imutils
pip install numpy
pip install opencv-python
pip install sklearn
pip install dlib
```
## Usage
Run the server py executing the following command in the project directory:

`python manage.py runserver`

After that, the server is listening on port 8000 waiting for POST requests to one of the following routes:

`/face/storetrain` to store base64 encoded images locally and re-train the neuronal network for the respective smart mirror.

`/face/recognizeimage` to ask the neuronal network to try and detect and recognize a face on a base64 encoded image.

For details about what to send with the request, inspect the code for the respective handlers in [api.py](./facerecognition/api.py).
Documentation might come shortly :)

## Contributors
This project is developed and maintain by [Emre Besogul](https://github.com/emrebesogul) and [Felix Waldbach](https://github.com/felixwaldbach). 
For questions reach out to us!
