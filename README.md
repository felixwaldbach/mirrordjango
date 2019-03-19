# Mirrordjango

## What does this Repository contain?
This repository holds code for a basic [Django Python Webserver](https://www.djangoproject.com/) using a REST API. 
The api serves with routes to trigger face recognition algorithms. For details about these algorithms see [Adrian Rosebrock's 
PyImageSearch](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) blog.
Most of the code used in this approach was inspired by his blog and is working on the same principles. Thank you, Adrian!

This project was created to perform face recognition in order to authenticate user's in a [selfmade smart mirror application](https://github.com/felixwaldbach/mirrorserver) we are
currently building for university. However, since the smart mirror application itself is developed for and running on a RaspberryPi with limited processing power, the face recognition
is performed externally on this little server. This project has only been tested on a 2018 MacBook Pro so far.

## Getting started
First of all, ensure to have a running python instance ready on your machine. We used an [anaconda virtual environment](https://www.anaconda.com/) with Python 3.6 installed. We highly recommend to set up one up if you haven't already.
To execute the project, install following packages brew:
```
brew install cmake
brew install boost
```
Then, install following packages using pip:
```
pip install django
pip install django-cors-headers
pip install imutils
pip install numpy
pip install opencv-python
pip install sklearn
pip install dlib
```

To execute the face recognition algorithms, a facial landmark file is needed in the root directory of this project. It can be downloaded together with the source code of [Adrian Rosebrock's blog post on face alignment](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/). It's called `shape_predictor_68_face_landmarks.dat` Check out the post if you are there already :) 

Furthermore, for better results we suggest including an unknown face dataset inside the project. Take for instance 20 images of random people, put them inside a folder named `unknown` and store the folder inside `./facerecognition` folder.

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
