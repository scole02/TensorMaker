import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from io import BytesIO
import base64
from PIL import Image as im
#from .forms import NameForm
# from keras.applications import vgg16
# from keras.applications.imagenet_utils import decode_predictions
# from keras.preprocessing.image import img_to_array, load_img
# from tensorflow.python.keras.backend import set_session


from core.train import start_training


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     mpld3.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     mpld3.imshow(inp)

def data(request):
    if request.method == "POST":
        start_training(request)
        #
        # Django image API 
        #


        # inp = out.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)
        # data = im.fromarray(inp)
        # data.save('test.png')

    return render(request, "data.html")

def test(request):
    if request.method == "POST":
        #
        # Django image API
        #

        return render(request, "test.html")

    else:
        return render(request, "test.html")
    
    return render(request, "test.html")





