from django.http import HttpResponse
import json
import base64
import os

# f gets closed when you exit the with statement
# Now save the value of filename to your database

# Create your views here.
from facerecognition.facerecognitionalgorithms import extract_embeddings, train_model, recognize_image, align_faces


def handle_recognize_image(request):
    payload = json.loads(request.body.decode("utf-8"))
    if os.path.isdir("./facerecognition/output/" + payload['mirror_uuid']):
        recognize_payload = {
            'recognizer': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload['mirror_uuid'] +
                          "-recognizer.pickle",
            'le': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload['mirror_uuid'] +
                  "-le.pickle",
            'image_base64': payload['image_base64'],
            'mirror_uuid': payload['mirror_uuid']
        }
        response = recognize_image(recognize_payload)
        return HttpResponse(json.dumps(response))
    else:
        with open('responseMessages.json', 'r') as responseMessages:
            responseMessage = json.load(responseMessages)
            return HttpResponse(json.dumps({
                'status': True,
                'message': responseMessage['NO_MIRROR_DIRECTORY']
            }))


def handle_store(request):
    if request.method == 'POST':
        payload = json.loads(request.body.decode("utf-8"))
        if not os.path.isdir("./facerecognition/output/" + payload['mirror_uuid'] + '/' + payload['userId']):
            os.makedirs("./facerecognition/output/" + payload['mirror_uuid'] + '/' + payload['userId'])

        print('Processing')
        imgdata = base64.b64decode(payload['base64'].replace('data:image/png;base64,', ''))
        filename = './facerecognition/output/' + payload['mirror_uuid'] + '/' + payload['userId'] + '/' + payload[
            'filename']  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            print('Saving file ' + filename)
            f.write(imgdata)
            response = align_faces({
                'image': filename
            })
        if payload['last_image']:
            embeddings_payload = {
                'dataset': "./facerecognition/output/" + payload['mirror_uuid'],
                'embeddings': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload[
                    'mirror_uuid'] + ".pickle"
            }
            response = extract_embeddings(embeddings_payload)
            try:
                response = handle_train(payload['mirror_uuid'])
                response['last_image'] = payload['last_image']
                return HttpResponse(json.dumps(response))
            except Exception as e:
                return HttpResponse(json.dumps({
                    'status': False,
                    'message': e
                }))
        else:
            return HttpResponse(json.dumps({
                'status': True,
                'message': "Image saved successfully"
            }))


def handle_train(mirror_uuid):
    train_payload = {
        'embeddings': "./facerecognition/output/" + mirror_uuid + "/" + mirror_uuid + ".pickle",
        'recognizer': "./facerecognition/output/" + mirror_uuid + "/" + mirror_uuid +
                      "-recognizer.pickle",
        'le': "./facerecognition/output/" + mirror_uuid + "/" + mirror_uuid +
              "-le.pickle"
    }
    try:
        response = train_model(train_payload)
        return {
            'status': True,
            'message': response
        }
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': e
        }
