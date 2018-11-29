import time
import numpy as np

import workflow as wf
#from workflow import wf.WFException

class AccumulatedValue(object):

    def __init__(self, name, avgWidth=2):
        self.name = name

        self.acc = []
        self.stamp = []

        if (avgWidth <= 0):
            exp = wf.WFException(
                "Averaging width must be a positive integer.", "AccumulatedValue")
            raise(exp)

        self.avg = []
        self.avgWidth = avgWidth
        self.avgCount = 0

        self.xLabel = "Stamp"
        self.yLabel = "Value"

    def push_back(self, v, stamp=None):
        self.acc.append(v)

        if (stamp is not None):
            self.stamp.append(stamp)
        else:
            if (0 == len(self.stamp)):
                self.stamp.append(0)
            else:
                self.stamp.append(self.stamp[-1] + 1)

        # Calculate the average.
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

    def push_back_array(self, a, stamp=None):
        nA = len(a)
        nS = 0

        if (stamp is not None):
            nS = len(stamp)

            if (nA != nS):
                # This is an error.
                desc = """Lengh of values should be the same with the length of the stamps.
                len(a) = %d, len(stamp) = %d.""" % (nA, nS)
                exp = wf.WFException(desc, "push_back_array")
                raise(exp)

            for i in range(nA):
                self.push_back(a[i], stamp[i])
        else:
            for i in range(nA):
                self.push_back(a[i])

    def clear(self):
        self.acc = []
        self.stamp = []
        self.avg = []

    def last(self):
        if (0 == len(self.acc)):
            # This is an error.
            desc = "The length of the current accumulated values is zero"
            exp = wf.WFException(desc, "last")
            raise(exp)

        return self.acc[-1]

    def last_avg(self):
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
        print("acc: ")
        print(self.acc)
        print("stamp: ")
        print(self.stamp)

    def dump(self, outDir, prefix="", suffix=""):
        # Convert acc and avg into NumPy arrays.
        acc = np.column_stack((np.array(self.stamp).astype(
            np.float), np.array(self.acc).astype(np.float)))
        avg = np.column_stack((np.array(self.stamp).astype(
            np.float), np.array(self.avg).astype(np.float)))

        # Dump the files.
        np.save(outDir + "/" + prefix + self.name + suffix + ".npy", acc)
        np.savetxt(outDir + "/" + prefix + self.name + suffix + ".txt", acc)

        # np.save(    outDir + "/" + prefix + self.name + suffix + "_avg.npy", avg )
        # np.savetxt( outDir + "/" + prefix + self.name + suffix + "_avg.txt", avg )
