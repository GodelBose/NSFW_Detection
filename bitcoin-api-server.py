from flask import Flask
from flask import request
from two1.wallet import Wallet
from two1.bitserv.flask import Payment
import sys
import os
import requests
from keras.preprocessing.image import ImageDataGenerator
import json
import datetime
import shutil
import numpy as np

now = datetime.datetime.now()
try:
    model_weights= sys.argv[1]
except:
    raise Exception('No model weights were provided.')
app = Flask(__name__)
wallet = Wallet()
payment = Payment(app, wallet)
img_channels = 3
width = 224
height = 224
nb_dense = 4000

model = Detector((height, width, img_channels), nb_dense)
model.load_weights(model_weights)

path = os.getcwd()
f = open('logger.txt','a')
f.write(str(now)+'\n')
f.close()
test_mode = False

@app.route('/nsfw_detection')
@payment.required(100)
def nsfw_detection():
    now = datetime.datetime.now()
    # log time of user requests
    f = open('logger.txt','a')
    f.write(' ' + str(now)+'\n')
    f.close()
    counter = 1
    if not test_mode:
        text = str(request.args.get('url'))
    else:
        text = URL
    if not os.path.exists(path+'/temp'):
        os.mkdir(path+'/temp')
        os.mkdir(path+'/temp/pic')
    try:
        response = requests.get(text, stream=True)
        f = open(path+'/temp/pic/'+str(counter)+'.jpg',"wb")
        f.write(response.content)
        f.close()
    except:
        raise FileNotFoundError('Received 404 when fetching '+text)
    counter += 1
    test_datagen = ImageDataGenerator()
    data = {}
    for x,y in test_datagen.flow_from_directory(path+'/temp',target_size=(height,width), batch_size=1, save_to_dir=os.getcwd()+'/uploaded', save_prefix='gen_pic_'):
        prediction = model.predict(x.astype(np.float32))
        break
    shutil.rmtree(path+'/temp')
    data['Nsfw'] = 'True' if prediction>= 0.5 else 'False'

    return json.dumps(data)

# Initialize and run the server
if __name__ == '__main__':
    app.run(host='::', port=5000)
