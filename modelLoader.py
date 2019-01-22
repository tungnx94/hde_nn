from network import MobileReg, MobileEncoderReg

Thresh = 0.005  # unlabel_loss threshold
    
class ModelLoader(object):

    def __init__(self):
        self.name = "Model-Loader"

    def load(self, modelType, mobileNet=None):
        if modelType == 0:
            model = MobileReg(lamb=0.1, thresh=Thresh)
        elif modelType == 1:
            model = MobileEncoderReg(lamb=0.001)

        if mobileNet is not None:
            model.load_mobilenet(mobileNet)
            print "Loaded MobileNet ", mobileNet

        return model

    def load_trained(self, modelType, trained_params):
        model = self.load(modelType) 
        model.load_pretrained(trained_model)

        print "Loaded weights from ", trained_params

        return model