import os
import numpy as np
import pandas as pd
import workflow as wf


class AccumulatedValue(object):

    def __init__(self, avgWidth):
        if (avgWidth <= 0):
            exp = wf.WFException(
                "Averaging width must be a positive integer.", "AccumulatedValue")
            raise(exp)

        self.avList = list(avgWidth.keys())
        self.avgWidth = avgWidth  # map an av to its window size
        self.stamp = []  # time stamps

        self.avgCount = {av:0 for av in self.avList}  
        self.acc = {av:[] for av in self.avList}   # values
        self.avg = {av:[] for av in self.avList}   # average values

    def keys(self):
        return self.avList 

    def push_value(self, name, value, stamp=0):
        if (self.stamp == []) or ((self.stamp != []) and (self.stamp[-1] != stamp)):
            self.stamp.append(stamp) 

        self.acc[name].append(value)
        self.avgCount[name] = self.push_avg(value, self.acc[name], self.avg[name], self.avgCount[name], self.avgWidth[name]) 

    def push_avg(self, value, acc, avg, avgCount, avgWidth):
        if avgCount == 0:
            avg.append(value)
            avgCount = 1
        else:
            # calculate new average
            if avgCount < avgWidth:
                new_avg = (avg[-1]*avgCount + value) / (avgCount + 1)
            else:
                new_avg = (avg[-1]*avgCount - acc[-1-avgCount] + value) / avgCount
            avg.append(new_avg)
            avgCount = min(avgCount + 1, avgWidth)

        return avgCount 

    def push_stamp(self, new_idx):
        self.stamp.append(new_idx)

    def clear(self):
        """ reset all values """
        self.acc = {}
        self.stamp = {}
        self.avg = {}

    def last(self, name):
        """ return last value """
        if (0 == len(self.acc[name])):
            # This is an error.
            desc = "The length of the current accumulated values is zero"
            exp = wf.WFException(desc, "last")
            raise(exp)

        return self.acc[name][-1]

    def last_avg(self, name):
        """ return last average value """
        if (0 == len(self.avg[name])):
            # This is an error.
            desc = "The length of the current accumulated values is zero"
            exp = wf.WFException(desc, "last_avg")
            raise(exp)

        return self.avg[name][-1]

    def get_num_values(self):
        return len(self.acc)

    def get_stamps(self):
        return self.stamp

    def get_values(self, name):
        return self.acc[name]

    def get_avg_values(self, name):
        return self.avg[name]

    def save_csv(self, outDir):
        # save all avs and averages to a CSV file 
        data_dict = {"stamp": self.stamp} 
        for av in self.avList:
            data_dict[av] = self.acc[av]
            data_dict[av + '_avg'] = self.avg[av]

        save_path = os.path.join(outDir, "values.csv")
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(save_path, index=False)

    def show_raw_data(self):
        # for debugging purpose
        print("stamp: ", self.stamp)
        print("acc: ", self.acc)
        print("avg: ", self.avg)
