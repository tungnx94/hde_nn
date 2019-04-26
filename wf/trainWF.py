import sys
sys.path.insert(0, '..')

import os
import torch
import numpy as np 
import torch.optim as optim 

from datetime import datetime
from .workflow import WorkFlow
import utils
import copy
import math

# TODO: save best result 
class TrainWF(WorkFlow):

    def __init__(self, config):
        # create folders
        t = datetime.now().strftime("%m-%d_%H-%M")
        
        self.modelName = config["model"]["name"]

        workingDir = os.path.join("./log/train", config["dir"], t + "_" + self.modelName)
        #workingDir = os.path.join(config["dir"], t + "_" + self.modelName)

        self.traindir = os.path.join(workingDir, "train")
        self.modeldir = os.path.join(workingDir, "models")
        
        self.logdir = self.traindir
        self.logfile = config["log"]

        self.saveFreq = config["save_freq"]
        self.showFreq = config["show_freq"]
        self.valFreq = config["val_freq"]
        self.trainStep = config["train_step"]

        self.lr = config["lr"]
        self.batch = config["batch"]

        if "batch_val" in config:
            self.batch_val = config["batch_val"]
        if "lamb" in config:
            self.lamb = lamb

        self.save_best = ("save_best" in config) and (config["save_best"] == True)
        self.best_val_loss = math.inf

        for folder in [workingDir, self.traindir, self.modeldir]:
            utils.create_folder(folder)

        WorkFlow.__init__(self, config)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_model = copy.deepcopy(self.model)

    def save_model(self):
        """ Save :param: model to pickle file (pkl) """
        model_path = os.path.join(self.modeldir, str(self.countTrain) + ".pkl")
        torch.save(self.model.state_dict(), model_path)

    def save_best_model(self):
        path = self.modeldir + "/best.pkl"
        torch.save(self.best_model.state_dict(), path)

    def save_result(self, res_file):
        losses = self.evaluate_final()
        res = {"train": losses[0], "val": losses[1]}
        path = self.logdir + "/" + res_file
        utils.write_json(res, path)                

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)    
        self.save_snapshot()

        for avp in self.AVP:
            avp.write_image_final(self.logdir)

        #Save final results
        self.logger.info("Saving final results")
        self.save_result("results.json")

        if self.save_best:
            self.model = self.best_model
            self.save_result("results_best.json")

        self.logger.info("Done")

    def save_snapshot(self):
        """ write accumulated values and save temporal model """
        self.save_accumulated_values()
        self.save_model()
        if self.save_best:
            self.save_best_model()

        self.logger.info("Saved snapshot")

    def run(self):
        """ train on all samples """
        self.logger.info("Started training")
        # WorkFlow.train(self)

        self.model.train()
        for iteration in range(1, self.trainStep + 1):
            self.check_signal()

            self.countTrain += 1
            train_sample = self.next_train_sample()

            # backward pass
            train_error = self.train_error(train_sample)
            self.optimizer.zero_grad()
            train_error.backward()
            self.optimizer.step()

            # Validation step
            if iteration % self.valFreq == 0:
                self.check_signal()
                self.model.eval()

                if self.save_best:
                    self.evaluate_sample_set(train_sample)
                else:
                    val_sample = self.next_val_sample()
                    self.evaluate_sample(train_sample, val_sample)

                self.model.train()

            # output screen
            if iteration % self.showFreq == 0:
                self.logger.info("#%6d %s" % (self.countTrain, self.get_log_str_avg()))

            # save temporary model
            if iteration % self.saveFreq == 0:
                self.save_snapshot()

        self.logger.info("Finished training")

    def evaluate_sample(self, train_sample, val_sample):
        """ update val loss history """
        train_losses = self.val_metrics(train_sample)
        val_losses = self.val_metrics(val_sample)

        losses = np.concatenate((train_losses, val_losses))
        for idx, av in enumerate(self.config["losses"]):
            self.AV.push_value(av, losses[idx], self.countTrain)

    def evaluate_sample_set(self, train_sample):
        train_losses = self.val_metrics(train_sample)
        val_losses = self.evaluate_set(self.val_loader, self.next_val_sample)

        losses = np.concatenate((train_losses, val_losses))
        for idx, av in enumerate(self.config["losses"]):
            self.AV.push_value(av, losses[idx], self.countTrain)

        if self.best_val_loss > val_losses[0]:
            self.best_val_loss = val_losses[0]
            self.best_model = copy.deepcopy(self.model)


    def evaluate_set(self, loader, next_sample_func):
        loader.reset_iteration()
        loss_list = []
        n = loader.max_iteration()

        for i in range(n):
            sample = next_sample_func()
            loss = self.val_metrics(sample)
            loss_list.append(loss)

        losses = np.mean(np.array(loss_list), axis=0)
        return losses

    def evaluate_final(self):
        train_loss_name, val_loss_name = utils.split_half(self.config["losses"])

        train_loss_values = self.evaluate_set(self.train_loader, self.next_train_sample)
        val_loss_values = self.evaluate_set(self.val_loader, self.next_val_sample)

        train_loss = dict(zip(train_loss_name, train_loss_values))
        val_loss = dict(zip(val_loss_name, val_loss_values))

        return (train_loss, val_loss)

    def next_train_sample(self):
        pass

    def next_val_sample(self):
        pass

    def train_error(self):
        pass

    def val_metrics(self, sample):
        pass
