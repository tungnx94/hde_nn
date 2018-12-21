import os
import torch

class HDENet(torch.nn.Module):

    def __init__(self, device=None):
        super(HDENet, self).__init__()
        self.countTrain = 0
        self.device = device

        if device is None:  #select default if not specified
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_to_device(self):
        self.to(self.device)

    def load_from_npz(self, file):
        model_dict = self.state_dict()

        preTrainDict = torch.load(file)
        preTrainDict = {k: v for k, v in preTrainDict.items() if k in model_dict}

        model_dict.update(preTrainDict)
        self.load_state_dict(model_dict)
        
        """
        print 'preTrainDict:',preTrainDict.keys()
        print 'modelDict:',model_dict.keys()
    
        for item in preTrainDict:
            print '  Load pretrained layer: ', item
        for item in model_dict:
            print '  Model layer: ',item
        """

    def load_pretrained(self, file):
        # file needs to point to a relative path
        modelname = os.path.splitext(os.path.basename(file))[0]
        self.countTrain = int(modelname.split('_')[-1])
        self.load_from_npz(file)