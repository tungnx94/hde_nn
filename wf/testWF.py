import os

from datetime import datetime
from workflow import WorkFlow
from dataset import DatasetLoader

class TestWF(WorkFlow):

    def __init__(self, config):
        t = datetime.now().strftime('%m-%d_%H:%M')

        workingDir = config['dir']
        self.modeldir = os.path.join(workingDir, 'models') 
        self.testdir = os.path.join(workingDir, 'test', config['prefix'] + "_" + t)

        self.saveFreq = config['save_freq']
        self.showFreq = config['show_freq']
        self.batch = config['batch']

        self.logdir = self.testdir
        self.logfile = config['log']

        config['model']['trained'] = os.path.join(self.modeldir, config['model']['trained'])

        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir)

        WorkFlow.__init__(self, config)

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_accumulated_values()

    def prepare_dataset(self, dloader):
        test_dts = self.load_dataset()
        self.test_loader = dloader.loader(test_dts, self.batch)

    def run(self):
        self.logger.info("Started testing")
        
        self.model.eval()

        WorkFlow.test(self)

        iteration = 0
        for sample_batch in self.test_loader:

            self.test(sample_batch)

            iteration += 1
            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))
            # save temporary values
            if iteration % self.saveFreq == 0:
                self.save_accumulated_values()

        self.logger.info("Finished testing")

    def test(self):
        pass