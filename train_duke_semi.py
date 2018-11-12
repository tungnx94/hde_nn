import sys
sys.path.append('../WorkFlow')

import config as cnf

from workflow import WorkFlow
from train_wf import TrainWF
from test_wf import TestFolderWF, TestLabelSeqWF, TestUnlabelSeqWF

train_label_file = '/datadrive/person/DukeMTMC/trainval_duke.txt'
test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'
unlabel_file = 'duke_unlabeldata.pkl'
saveModelName = 'facing'

test_label_img_folder = '/home/wenshan/headingdata/val_drone'
test_unlabel_img_folder = '/datadrive/exp_bags/20180811_gascola'

pre_mobile_model = 'pretrained_models/mobilenet_v1_0.50_224.pth'
pre_model = 'models/1_2_facing_20000.pkl'

TestType = 2  # 0: none, 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence
exp_prefix = 'vis_1_3_'


def select_WF(TestType):
    """ choose WF from test type """
    trainWF = TrainWF("./", prefix=exp_prefix,
                      mobile_model=pre_mobile_model, trained_model=pre_model)

    testLabelWF = TestLabelSeqWF("./", prefix=exp_prefix,
                                 mobile_model=pre_mobile_model, trained_model=pre_model)

    testFolderWF = TestFolderWF("./", prefix=exp_prefix,
                                mobile_model=pre_mobile_model, trained_model=pre_model)

    testUnlabelWF = TestUnlabelSeqWF("./", prefix=exp_prefix,
                                     mobile_model=pre_mobile_model, trained_model=pre_model)

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
