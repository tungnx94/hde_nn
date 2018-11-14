from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np

from visdom import Visdom


class AccumulatedValue(object):

    def __init__(self, name, avgWidth=2):
        self.name = name

        self.acc = []
        self.stamp = []

        if (avgWidth <= 0):
            exp = WFException(
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
                exp = WFException(desc, "push_back_array")
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
            desc = "The length of the current accumulated values is zero."
            exp = WFException(desc, "last")
            raise(exp)

        return self.acc[-1]

    def last_avg(self):
        if (0 == len(self.avg)):
            # This is an error.
            desc = "The length of the current accumulated values is zero."
            exp = WFException(desc, "last")
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


class VisdomLinePlotter(AccumulatedValuePlotter):
    # Class/Static variables.
    vis = None
    visStartUpSec = 1

    def __init__(self, name, av, avNameList, avAvgFlagList=None, semiLog=False):
        super(VisdomLinePlotter, self).__init__(
            name, av, avNameList, avAvgFlagList)

        self.count = 0
        self.minPlotPoints = 2

        self.visLine = None

        if (True == semiLog):
            self.plotType = "log"
        else:
            self.plotType = "linear"

    def initialize(self):
        if (VisdomLinePlotter.vis is not None):
            print("visdom already initialized.")
            return

        VisdomLinePlotter.vis = Visdom(
            server='http://localhost', port=8097, use_incoming_socket=False)

        while not VisdomLinePlotter.vis.check_connection() and VisdomLinePlotter.visStartUpSec > 0:
            time.sleep(0.1)
            VisdomLinePlotter.visStartUpSec -= 0.1
        assert VisdomLinePlotter.vis.check_connection(
        ), 'No connection could be formed quickly'

        print("VisdomLinePlotter initialized.")

    def get_vis(self):
        return VisdomLinePlotter.vis

    def is_initialized(self):
        vis = self.get_vis()

        if (vis is None):
            return False
        else:
            return True

    def update(self):
        # Check if Visdom is initialized.
        if (False == self.is_initialized()):
            exp = WFException("Visdom has not been initialized yet.", "update")
            raise(exp)

        # Gather the data.
        # nLines = len( self.avNameList )
        nMaxPoints = 0

        for name in self.avNameList:
            # Find the AccumulatedVariable object.
            av = self.AV[name]
            nPoints = av.get_num_values()

            if (nPoints > nMaxPoints):
                nMaxPoints = nPoints

        if (nMaxPoints < self.minPlotPoints):
            # Not enough points to plot, do nothing.
            return

        # Enough points to plot.
        # Get the points to be ploted.
        nameList = []
        for name in self.avNameList:
            av = self.AV[name]
            lastIdx = self.plotIndexDict[name]
            pointsInAv = av.get_num_values()

            if (pointsInAv - 1 > lastIdx and 0 != pointsInAv):
                nameList.append(name)

        if (0 == len(nameList)):
            # No update actions should be performed.
            return

        # Retreive the Visdom object.
        vis = self.get_vis()

        for i in range(len(nameList)):
            name = nameList[i]
            av = self.AV[name]
            lastIdx = self.plotIndexDict[name]

            x = np.array(av.get_stamps()[lastIdx + 1:])

            if (self.visLine is None):
                # Create the Visdom object.
                self.visLine = vis.line(
                    X=x,
                    Y=np.array(av.get_values()[lastIdx + 1:]),
                    name=name,
                    opts=dict(
                        showlegend=True,
                        title=self.title,
                        xlabel=self.xlabel,
                        ylabel=self.ylabel,
                        ytype=self.plotType,
                        margintop=30
                    )
                )
            else:
                # Append data to self.visLine.
                vis.line(
                    X=x,
                    Y=np.array(av.get_values()[lastIdx + 1:]),
                    win=self.visLine,
                    name=name,
                    update="append",
                    opts=dict(
                        showlegend=True
                    )
                )

        for i in range(len(nameList)):
            name = nameList[i]
            av = self.AV[name]
            lastIdx = self.plotIndexDict[name]

            # Average line.
            if (True == self.avAvgFlagDict[name]):
                vis.line(
                    X=np.array(av.get_stamps()[lastIdx + 1:]),
                    Y=np.array(self.AV[name].get_avg()[
                               self.plotIndexDict[name] + 1:]),
                    win=self.visLine,
                    name=name + "_avg",
                    update="append",
                    opts=dict(
                        showlegend=True
                    )
                )

            # Update the self.plotIndexDict.
            self.plotIndexDict[name] = self.AV[name].get_num_values() - 1

        self.count += 1
