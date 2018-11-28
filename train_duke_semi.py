import config as cnf

from workflow.train_wf import TrainWF
from workflow.wtest_wf import TestFolderWF, TestLabelSeqWF, TestUnlabelSeqWF

PreMobileModel = 'network/pretrained_models/mobilenet_v1_0.50_224.pth'

#PreModel = 'data/models/1_2_facing_20000.pkl'
PreModel = None

TestType = 0  # 0: none(train), 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence
ExpPrefix = 'vis_1_3_'


def select_WF(TestType):
    """ choose WF from test type """
    trainWF = TrainWF("./", prefix=ExpPrefix,
                      mobile_model=PreMobileModel, trained_model=PreModel)

    testLabelWF = TestLabelSeqWF("./", prefix=ExpPrefix,
                                 mobile_model=PreMobileModel, trained_model=PreModel)

    testFolderWF = TestFolderWF("./", prefix=ExpPrefix,
                                mobile_model=PreMobileModel, trained_model=PreModel)

    testUnlabelWF = TestUnlabelSeqWF("./", prefix=ExpPrefix,
                                     mobile_model=PreMobileModel, trained_model=PreModel)

    wfs = [trainWF, testLabelWF, testFolderWF, testUnlabelWF]

    return wfs[TestType]


def main():
    """ Train and validate new model """
    try:
        # Instantiate workflow.
        wf = select_WF(TestType)

        wf.initialize()
        wf.run()
        wf.finalize()

    except WorkFlow.SigIntException as e:
        wf.finalize()
    except WorkFlow.WFException as e:
        print(e.describe())

if __name__ == "__main__":
    main()
