import config as cnf

from wf import WFException, TrainSSWF, TestFolderWF, TestLabelSeqWF, TestUnlabelSeqWF

# 0: none(train), 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence
TestType = 0

# 0: MobileReg, 1: MobileEncoderReg, 3: Vanilla ?
ModelType = 1

PreMobileModel = 'network/pretrained_models/mobilenet_v1_0.50_224.pth'
PreModel = './log/sample_12-21_12:11/models/facing_11500.pkl' # 
# PreModel = None
ExpPrefix = 'sample'  # model name, should be unique

# TestFolder = "./log/sample_12-18_17:41"
# TestFolder = ./log/sample_12-21_01:55())
# TestModel = 'facing_37.pkl'#

def select_WF(TestType):
    """ choose WF from test type """
    # ugly code to avoid multiple instance of logger in WorkFlow
    if TestType == 0:
        return TrainSSWF("./log", ExpPrefix, ModelType, mobile_model=PreMobileModel, trained_model=PreModel)
    elif TestType == 1:
        return TestLabelSeqWF(TestFolder, "labelseq", ModelType, TestModel)
    elif TestType == 2:
        return TestFolderWF(TestFolder, "folder", ModelType, TestModel)
    else:  # 3
        return TestUnlabelSeqWF(TestFolder, "unlabelseq", ModelType, TestModel)


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
