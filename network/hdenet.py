import torch
import torch.nn as nn

class HDENet(nn.Module):

    def load_from_npz(self, params):
        model_dict = self.state_dict()

        preTrainDict = torch.load(params)
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