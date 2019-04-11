from .hdeReg import HDEReg
from .mobileRNN import MobileRNN
from .mobileReg import MobileReg
from .hdeRNN import HDE_RNN

Thresh = 0.005  # unlabel_loss threshold

class ModelLoader(object):

    def __init__(self):
        self.name = "Model-Loader"

    def load(self, config):
        model = None

        mtype = config["type"]
        mtrained = config["trained"]

        if mtype == 0:
            model = HDEReg(config)
        elif mtype == 1:
            model = MobileRNN(config)
        elif mtype == 2:
            model = MobileReg(config)
        elif mtype == 3:
            model = HDE_RNN(config)

        if mtrained is not None: # TODO: fix path 
            model.load_pretrained(mtrained)
            print("Loaded weights from", mtrained) # TODO: put to logger

        return model