import config as cnf

from wf import WFException, TrainWF, TestFolderWF, TestLabelSeqWF, TestUnlabelSeqWF

PreMobileModel = 'network/pretrained_models/mobilenet_v1_0.50_224.pth'

#PreModel = 'data/models/1_2_facing_20000.pkl'
PreModel = None

# 0: none(train), 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence
TestType = 0
ExpPrefix = 'sample'  # model name, should be unique


def select_WF(TestType):
    """ choose WF from test type """
    # ugly code to avoid multiple instance of logger in WorkFlow
    if TestType == 0:
        return TrainWF("./log", ExpPrefix, mobile_model=PreMobileModel, trained_model=PreModel)
    elif TestType == 1:
        return TestLabelSeqWF("./log", ExpPrefix, mobile_model=PreMobileModel, trained_model=PreModel)
    elif TestType == 2:
        return TestFolderWF("./log", ExpPrefix, mobile_model=PreMobileModel, trained_model=PreModel)
    else:  # 3
        return TestUnlabelSeqWF("./log", ExpPrefix, mobile_model=PreMobileModel, trained_model=PreModel)


def main():
    """ Train and validate new model """
    try:
        # Instantiate workflow.
        wf = select_WF(TestType)

        wf.initialize()
        wf.run()
        wf.finalize()

    except WFException as e:
        print "exception", e
        wf.finalize()

if __name__ == "__main__":
    main()
