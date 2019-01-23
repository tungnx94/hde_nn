import os

from datetime import datetime
from workflow import WorkFlow

class TestWF(WorkFlow):

    def __init__(self, config):
        t = datetime.now().strftime('%m-%d_%H:%M')

        workingDir = config['dir']
        self.modeldir = os.path.join(workingDir, 'models') 
        self.testdir = os.path.join(workingDir, 'test', config['prefix'] + "_" + t)

        self.testStep = cnf['test_step']
        self.saveFreq = cnf['save_freq']
        self.showFreq = cnf['show_freq']
        self.batch = cnf['batch']

        self.logdir = self.testdir
        self.logfile = cnf['log']

        self.modeldir = os.path.join(workingDir, 'models')
        self.modelName = cnf['model']['name']

        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir)

        WorkFlow.__init__(self, config)

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_accumulated_values()

    def prepare_dataset(self):
        test_dts = self.load_dataset()
        self.test_loader = d_loader.loader(test_dts, self.batch)

    def run(self):
        self.logger.info("Started testing")
        
        self.model.eval()
        for iteration in range(1, self.testStep + 1):
            WorkFlow.test(self)
            self.test()

            # output screen
            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary values
            if iteration % self.saveFreq == 0:
                self.save_accumulated_values()

        self.logger.info("Finished testing")

    def test(self):
        pass