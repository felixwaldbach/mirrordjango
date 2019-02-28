# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/adrian

# import the necessary packages
import json

from imutils.video import VideoStream
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
from imutils import paths
import imutils
import time
import pickle
import cv2
import os
import base64
import os.path
from imutils.face_utils import FaceAligner
import dlib


def recognize_image(payload):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["./face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["./face_detection_model",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(payload["recognizer"], "rb").read())
    le = pickle.loads(open(payload["le"], "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    imgdata = base64.b64decode(payload['image_base64'].replace('data:image/png;base64,', ''))
    filename = './facerecognition/output/' + payload['mirror_uuid'] + '/' + 'temporary.png'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    align_faces({
        'image': filename
    })
    image = cv2.imread(filename)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            print(text)

        # if os.path.isfile(filename):
    #     #   os.remove(filename)
    # for ad in all_detections:
    #     if not best_detection or ad['oc'] > best_detection['oc']:
    #         best_detection = ad
    with open('responseMessages.json', 'r') as responseMessages:
        responseMessage = json.load(responseMessages)
        return {
            'status': True,
            'message': responseMessage['RECOGNIZE_IMAGE_SUCCESS']
            # 'name': best_detection['name'],
            # 'proba': best_detection['oc']
        }


def extract_embeddings(payload):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    confidence = 0.5
    protoPath = os.path.sep.join(["./face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["./face_detection_model",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    var_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(payload["dataset"]))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        var_detector.setInput(imageBlob)
        detections = var_detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            var_confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if var_confidence > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(payload["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("[INFO] File closed")


def recognize_video(payload):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([payload["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([payload["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(payload["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(payload["recognizer"], "rb").read())
    le = pickle.loads(open(payload["le"], "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:  # payload["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    vs.stop()


def train_model(payload):
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(payload["embeddings"], "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    print("So far so good")
    # write the actual face recognition model to disk
    f = open(payload["recognizer"], "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    print("Opening le")
    # write the label encoder to disk
    f = open(payload["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()
    with open('responseMessages.json', 'r') as responseMessages:
        print("Sending response")
        responseMessage = json.load(responseMessages)
        return {
            'status': True,
            'message': responseMessage['TRAIN_MODEL_SUCCESS']
        }


def align_faces(payload):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")

    image = cv2.imread(payload['image'])
    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    # loop over the face detections
    for rect in rects:
        print("Face found")
        faceAligned = fa.align(image, gray, rect)
        cv2.imwrite(payload['image'], faceAligned)
    # os.remove(imagePath)
    return {
        'status': True,
        'message': 'Faces aligned'
    }
