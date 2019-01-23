from wf import WFException, TrainSSWF, TestLabelWF, TestUnlabelWF, TestLabelSeqWF
# from wf import *
from utils import ModelLoader, DatasetLoader

ExpPrefix = 'sample'  # model name, should be unique

# 0: none(train), 1: labeled image, 2: unlabeled sequence, 3: labeled seq
TestType = 0

# 0: Vanilla, 1: MobileRNN, 2: MobileReg, 3: MobileEncoderReg
ModelType = 2

MobileModel = 'network/pretrained_models/mobilenet_v1_0.50_224.pth'
MobileModel = None

# TrainedModel = './log/sample_12-21_12:11/models/facing_11500.pkl' # 
TrainedModel = None

Mean = [0.485, 0.456, 0.406]
Std = [0.229, 0.224, 0.225]

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1

TestBatch = 64
ValBatch = 0 #

d_loader = DatasetLoader(Mean, Std)
m_loader = ModelLoader()

class TrainDuke(TrainSSWF):

    def load_model(self):
        if TrainedModel is not None:
            return m_loader.load_trained(ModelType, TrainedModel)
        else:
            return m_loader.load(ModelType, MobileModel)

    def load_dataset(self):
        train_duke = d_loader.single_label('train-duke', 'DukeMTMC/train/train.csv')
        train_virat = d_loader.single_label('train-virat', 'VIRAT/person/train.csv')
        train_manual = d_loader.single_label('train-handlabel', 'handlabel/person.csv')

        #train_duke.resize()
        #train_virat.resize()
        #train_manual.resize()

        label_dts = d_loader.mix('Training-label', [train_duke, train_virat, train_manual], None)
        self.train_loader = d_loader.loader(label_dts, Batch)

        unlabel_duke = d_loader.folder_unlabel('duke-unlabel', 'DukeMTMC/train/images_unlabel')
        unlabel_ucf = d_loader.folder_unlabel('ucf-unlabel', 'UCF')
        unlabel_drone = d_loader.folder_unlabel('drone-unlabel', 'DRONE_seq') 
        #unlabel_duke.resize()
        #unlabel_ucf.resize()
        #unlabel_drone.resize()

        unlabel_dts = d_loader.mix('Training-unlabel', [unlabel_duke, unlabel_ucf, unlabel_drone])
        self.train_unlabel_loader = d_loader.loader(unlabel_dts, UnlabelBatch)

        val_dts = d_loader.duke_seq('val-dukeseq', 'DukeMTMC/train/val.csv', SeqLength)
        self.val_loader = d_loader.loader(val_dts, 1)

TestModelType = 0
TestFolder = "./log/sample_"
TestModel = 'facing_37.pkl'#

class TestDuke_1(TestLabelWF):
    def load_dataset(self):
        test_dts = d_loader.single_label('DRONE_test', 'DRONE_label', data_aug=False)
        self.test_loader = d_loader.loader(test_dts, TestBatch)

class TestDuke_2(TestUnlabelWF):
    def load_dataset(self):
        test_dts = d_loader.folder_unlabel('DRONE-seq', 'DRONE_seq', data_aug=False)
        self.test_loader = d_loader.loader(test_dts, 1)

class TestDuke_3(TestLabelSeqWF):
    def load_dataset(self):
        self.test = d_loader.duke_seq('DUKE-test', 'DukeMTMC/test/test.csv', SeqLength, data_aug=False)
        self.test_loader = d_loader.loader(test_dts, 1)

def select_WF(TestType):
    """ choose WF from test type """
    # avoid multiple instance of logger in WorkFlow
    if TestType == 0:
        return TrainDuke("./log", ExpPrefix)
    elif TestType == 1:
        return TestDuke_1(TestFolder, "labelseq", TestModelType, TestModel)
    elif TestType == 2:
        return TestDuke_2(TestFolder, "folder", TestModelType, TestModel)
    else:  # 3
        return TestDuke_3(TestFolder, "unlabelseq", TestModelType, TestModel)


def main():
    """ Train and validate new model """
    try:
        # Instantiate workflow.
        wf = select_WF(TestType)
        wf.proceed()

    except WFException as e:
        print "exception", e
        wf.finalize()

if __name__ == "__main__":
    main()
