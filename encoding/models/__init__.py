from .model_zoo import get_model
from .model_store import get_model_file
from .resnet import *
from .cifarresnet import *
from .base import *
from .fcn import *
from .psp import *
from .efficientFCN import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'efficientfcn': get_efficientfcn,
    }
    return models[name.lower()](**kwargs)
