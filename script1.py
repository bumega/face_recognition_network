from flask import Flask, request, make_response
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = Flask(__name__)
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu').eval()
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
pdist = torch.nn.PairwiseDistance(p=2)
cred = credentials.Certificate('')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/registration', methods=['POST'])
def registration():
    file = request.files['photo']
    img = Image.open(file.stream).convert("RGB")
    with torch.no_grad():
        boxes, probs = mtcnn.detect(img)
        print(probs)
        if len(probs) > 1:
            res = "too many faces"
            response = make_response(res, 401)
            return response
        elif probs[0] == None:
            res = "no one face"
            response = make_response(res, 402)
            return response
        else:
            img_cr = mtcnn(img)
            vec = resnet(img_cr)
            people_ref = db.collection('userData')
            people = people_ref.stream()
            flag = True
            for person in people:
                dist = pdist(torch.tensor(list(map(float, person.to_dict()['biometry'][2:-3].split(',')))), vec)
                if dist < 1:
                    flag = False
                    break
            if flag:
                return vec.tolist()
            else:
                res =  "your face is already in the database"
                response = make_response(res, 403)
                return response
            
            
@app.route('/recognition', methods=['POST'])
def recognition():
    file = request.files['photo']
    img = Image.open(file.stream).convert("RGB")
    with torch.no_grad():
        boxes, probs = mtcnn.detect(img)
        if len(probs) > 1:
            res = "too many faces"
            response = make_response(res, 401)
            return response
        elif probs[0] == None:
            res = "no one face"
            response = make_response(res, 402)
            return response
        else:
            img_cr = mtcnn(img)
            vec = resnet(img_cr)
            people_ref = db.collection('userData')
            people = people_ref.stream()
            mn = 1000
            for person in people:
                dist = pdist(torch.tensor(list(map(float, person.to_dict()['biometry'][2:-3].split(',')))), vec)
                if dist < mn:
                    mn = dist
                    mn_id = person.id
            if mn < 1:
                return mn_id
            else:
                res =  "your face is missing from the database"
                response = make_response(res, 404)
                return response
            




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)