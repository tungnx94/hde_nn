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

        mtype = config['type']
        mmobile = config['mobile'] if 'mobile' in config else None
        mtrained = config['trained'] if 'trained' in config else None

        if mtype == 0:
            model = HDEReg(config["extractor"])
        elif mtype == 1:
            model = MobileRNN(config["extractor"])
        elif mtype == 2:
            model = MobileReg(extractor=config["extractor"], lamb=0.1, thresh=Thresh)
        elif mtype == 3:
            model = HDE_RNN(config["extractor"])
        
        if mmobile is not None:
            model.load_mobilenet(config['mobile'])
            print("Loaded MobileNet ", mmobile)

        if mtrained is not None:
            model.load_pretrained(mtrained)
            print("Loaded weights from", mtrained)

        return model