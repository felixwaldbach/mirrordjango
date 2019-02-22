from django.http import HttpResponse
import json
import base64
import os

# f gets closed when you exit the with statement
# Now save the value of filename to your database

# Create your views here.
from facerecognition.facerecognitionalgorithms import extract_embeddings, train_model, recognize_image


def handle_recognize_image(request):
    payload = json.loads(request.body.decode("utf-8"))
    print(payload)
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


def handle_store_and_train(request):
    if request.method == 'POST':
        payload = json.loads(request.body.decode("utf-8"))
        if not os.path.isdir("./facerecognition/output/" + payload['mirror_uuid'] + '/' + payload['user_id']):
            os.makedirs("./facerecognition/output/" + payload['mirror_uuid'] + '/' + payload['user_id'])
        for i in payload['images']:
            print('Processing')
            imgdata = base64.b64decode(i['base64'].replace('data:image/png;base64,', ''))
            filename = './facerecognition/output/' + payload['mirror_uuid'] + '/' + payload['user_id'] + '/' + i[
                'name']  # I assume you have a way of picking unique filenames
            with open(filename, 'wb') as f:
                f.write(imgdata)
    embeddings_payload = {
        'dataset': "./facerecognition/output/" + payload['mirror_uuid'],
        'embeddings': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload['mirror_uuid'] + ".pickle"
    }
    response = extract_embeddings(embeddings_payload)
    train_payload = {
        'embeddings': embeddings_payload['embeddings'],
        'recognizer': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload['mirror_uuid'] +
                      "-recognizer.pickle",
        'le': "./facerecognition/output/" + payload['mirror_uuid'] + "/" + payload['mirror_uuid'] +
              "-le.pickle"
    }
    try:
        response = train_model(train_payload)
    except:
        print('exception')
    return HttpResponse(json.dumps(response))
