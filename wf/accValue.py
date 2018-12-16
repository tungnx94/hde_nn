import os
import numpy as np
import pandas as pd
import workflow as wf


class AccumulatedValue(object):

    def __init__(self, name, avgWidth=2):
        if (avgWidth <= 0):
            exp = wf.WFException(
                "Averaging width must be a positive integer.", "AccumulatedValue")
            raise(exp)

        self.name = name
        self.avgWidth = avgWidth  # average window size
        self.avgCount = 0

        self.acc = []   # values
        self.avg = []   # average values
        self.stamp = []  # time stamps

    def push_back(self, v, stamp=None):
        self.acc.append(v)

        if (stamp is not None):
            self.stamp.append(stamp)
        else:
            if (0 == len(self.stamp)):
                self.stamp.append(0)
            else:
                self.stamp.append(self.stamp[-1] + 1)

        # Calculate new average.
        if (0 == len(self.avg)):
            self.avg.append(v)
            self.avgCount = 1
        else:
            if (self.avgCount < self.avgWidth):
                self.avg.append(
                    (self.avg[-1] * self.avgCount + self.acc[-1]) / (self.avgCount + 1))
                self.avgCount += 1
            else:
                self.avg.append(
                    (self.avg[-1] * self.avgCount - self.acc[-1 -
                                                             self.avgCount] + self.acc[-1]) / self.avgCount
                )

    def clear(self):
        """ reset all values """
        self.acc = []
        self.stamp = []
        self.avg = []

    def last(self):
        """ return last value """
        if (0 == len(self.acc)):
            # This is an error.
            desc = "The length of the current accumulated values is zero"
            exp = wf.WFException(desc, "last")
            raise(exp)

        return self.acc[-1]

    def last_avg(self):
        """ return last average value """
        if (0 == len(self.avg)):
            # This is an error.
            desc = "The length of the current accumulated values is zero"
            exp = wf.WFException(desc, "last_avg")
            raise(exp)

        return self.avg[-1]

    def get_num_values(self):
        return len(self.acc)

    def get_values(self):
        return self.acc

    def get_stamps(self):
        return self.stamp

    def get_avg(self):
        return self.avg

    def show_raw_data(self):
        print("%s" % (self.name))
        print("acc: ", self.acc)
        print("stamp: ", self.stamp)

    def save_csv(self, outDir):
        data_dict = {"stamp": self.stamp, "val": self.acc, "avg": self.avg}
        df = pd.DataFrame.from_dict(data_dict)

        save_path = os.path.join(outDir, self.name + ".csv")
        df.to_csv(save_path)
