from wf import *
from dataset import DatasetLoader
from utils import read_json

Mean = [0.485, 0.456, 0.406]
Std = [0.229, 0.224, 0.225]
d_loader = DatasetLoader(Mean, Std)

# 0: Vanilla, 1: MobileRNN, 2: MobileReg, 3: MobileEncoderReg
ModelType = 1

# TODO: pass as command parameter
TrainConfig = read_json('config/train1.json')
TestConfig = None

class TrainSSL(TrainSSWF):

    def load_dataset(self):
        """
        train_duke = d_loader.single_label('train-duke', 'DukeMTMC/train/train.csv')
        train_virat = d_loader.single_label('train-virat', 'VIRAT/person/train.csv')
        train_manual = d_loader.single_label('train-handlabel', 'handlabel/person.csv')
        #train_duke.resize()
        #train_virat.resize()
        #train_manual.resize()
        label_dts = d_loader.mix('Training-label', [train_duke, train_virat, train_manual], None)
        
        unlabel_duke = d_loader.folder_unlabel('duke-unlabel', 'DukeMTMC/train/images_unlabel')
        unlabel_ucf = d_loader.folder_unlabel('ucf-unlabel', 'UCF')
        unlabel_drone = d_loader.folder_unlabel('drone-unlabel', 'DRONE_seq') 
        #unlabel_duke.resize()
        #unlabel_ucf.resize()
        #unlabel_drone.resize()
        unlabel_dts = d_loader.mix('Training-unlabel', [unlabel_duke, unlabel_ucf, unlabel_drone])
        """

        label_dts = d_loader.single_label('train-duke', 'DukeMTMC/train/train.csv')
        unlabel_dts = d_loader.folder_unlabel('duke-unlabel', 'DukeMTMC/train/images_unlabel')
        val_dts = d_loader.duke_seq('val-dukeseq', 'DukeMTMC/train/val.csv', self.config['seq_length'])

        return (label_dts, unlabel_dts, val_dts)

class TrainVanilla(TrainSLWF):

    def load_dataset(self):
        train_duke = d_loader.single_label('train-duke', 'DukeMTMC/train/train.csv')
        val_duke = d_loader.single_label('val-duke', 'DukeMTMC/train/val.csv')

        return (train_duke, test_dts)

class TrainRNN(TrainRNNWF):

    def load_dataset(self):
        train_duke = d_loader.duke_seq('train-duke', 'DukeMTMC/train/train.csv', self.config['seq_length'])
        val_duke = d_loader.duke_seq('val-duke', 'DukeMTMC/train/val.csv', self.config['seq_length'])

        return (train_duke, test_dts)

class TestDuke_1(TestLabelWF):
    def load_dataset(self):
        return d_loader.single_label('DRONE_test', 'DRONE_label/test.csv', data_aug=False)

class TestDuke_2(TestUnlabelWF):
    def load_dataset(self):
        return d_loader.folder_unlabel('DRONE-seq', 'DRONE_seq', data_aug=False)

class TestDuke_3(TestLabelSeqWF):
    def load_dataset(self):
        return d_loader.duke_seq('DUKE-test', 'DukeMTMC/test/test.csv', TestConfig['seq_length'], data_aug=False)

def select_WF(TestType):
    # avoid multiple instance of logger in WorkFlow
    if TestType == 0:
        return TrainSSL(TrainConfig)
        # return TrainVanilla(TrainConfig)
        # return TrainRNN(TrainConfig)
    elif TestType == 1:
        TestConfig = read_json('config/test01.json')
        return TestDuke_1(TestConfig)
    elif TestType == 2:
        TestConfig = read_json('config/test02.json')
        return TestDuke_2(TestConfig)
    else:  # 3
        TestConfig = read_json('config/test03.json')
        return TestDuke_3(TestConfig)


def main():
    """ Train and validate new model """
    # 0: none(train), 1: labeled image, 2: unlabeled sequence, 3: labeled seq
    TestType = 0

    try:
        wf = select_WF(TestType)
        wf.prepare_dataset(d_loader)
        wf.proceed()

    except WFException as e:
        print "exception", e
        wf.finalize()

if __name__ == "__main__":
    main()
