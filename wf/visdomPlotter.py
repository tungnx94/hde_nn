import time

from visdom import Visdom
from workflow import WFException
from avPlotter import AccumulatedValuePlotter


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
