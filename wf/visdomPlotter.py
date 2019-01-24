import time
import numpy as np 

from visdom import Visdom
from exception import WFException
from avPlotter import AccumulatedValuePlotter


class VisdomLinePlotter(AccumulatedValuePlotter):
    # Class/Static variables.
    vis = None
    visStartUpSec = 1

    def __init__(self, name, av, avNameList, plot_average=True, semiLog=False):
        super(VisdomLinePlotter, self).__init__(
            name, av, avNameList, plot_average)

        self.count = 0
        self.minPlotPoints = 2
        self.visLine = None

        if (True == semiLog):
            self.plotType = "log"
        else:
            self.plotType = "linear"

    def initialize(self):
        # create new Vis instance
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
        nMaxPoints = self.AV.get_num_values()

        if (nMaxPoints < self.minPlotPoints):
            # Not enough points to plot, do nothing.
            return

        # Enough points to plot.
        vis = self.get_vis()

        stamps = av.get_stamps()
        x = np.array(stamps[self.lastIdx + 1:])
        for name in self.avNameList:
            y = np.array(self.AV.get_values(name)[self.lastIdx + 1:])
            if self.visLine is None:
                # Create the Visdom object.
                self.visLine = vis.line(
                    X=x,
                    Y=y,
                    name=name,
                    opts=dict(
                        showlegend=True,
                        title=self.name,
                        xlabel='iteration',
                        ylabel='loss',
                        ytype=self.plotType,
                        margintop=30
                    )
                )
            else:
                # Append data to self.visLine.
                vis.line(
                    X=x,
                    Y=y,
                    win=self.visLine,
                    name=name,
                    update="append",
                    opts=dict(
                        showlegend=True
                    )
                )

            # Average line.
            if self.plot_average:
                y = np.array(self.AV.get_avg_values(name)[self.lastIdx + 1:])
                vis.line(
                    X=x,
                    Y=y,
                    win=self.visLine,
                    name=name + "_avg",
                    update="append",
                    opts=dict(
                        showlegend=True
                    )
                )

        self.lastIdx = self.AV.get_num_values() - 1
