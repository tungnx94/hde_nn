import matplotlib.pyplot as plt

from workflow import WFException


class AccumulatedValuePlotter(object):

    def __init__(self, name, av, avNameList, avAvgFlagList=None):
        self.name = name
        self.AV = av
        self.avNameList = avNameList

        if len(self.avNameList) == 0:
            exp = WFException("The avNameList is empty.",
                              "AccumulatedValuePlotter")
            raise(exp)

        # Pack the avNameList in to index dictionary.
        initIndexList = [-1] * len(self.avNameList)

        self.plotIndexDict = dict(zip(self.avNameList, initIndexList))

        if avAvgFlagList is None:
            avAvgFlagList = [False] * len(self.avNameList)
        else:
            if len(self.avNameList) != len(avAvgFlagList):
                exp = WFException(
                    "The lenght of avAvgFlagList should be the same with avNameList", "AccumulatedValuePlotter")
                raise(exp)

        self.avAvgFlagDict = dict(zip(self.avNameList, avAvgFlagList))

        self.title = self.name
        self.xlabel = "xlabel"
        self.ylabel = "ylabel"

    def initialize(self):
        # The method of the base class cannot be invoked.
        exp = WFException(
            "initialize() of AccumulatedValuedPlotter base class could no be invoked directly.", "AccumulatedValuePlotter")
        raise(exp)

    def update(self):
        # The method of the base class cannot be invoked.
        exp = WFException(
            "update() of AccumulatedValuedPlotter base class could no be invoked directly.", "AccumulatedValuePlotter")
        raise(exp)

    def write_image(self, outDir, prefix="", suffix=""):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        legend = []

        for name in self.avNameList:
            av = self.AV[name]

            ax.plot(av.get_stamps(), av.get_values())
            legend.append(name)

        for name in self.avNameList:
            if (True == self.avAvgFlagDict[name]):
                av = self.AV[name]

                ax.plot(av.get_stamps(), av.get_avg())

                legend.append(name + "_avg")

        ax.legend(legend)
        ax.grid()
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        fig.savefig(outDir + "/" + self.title + suffix + ".png")
        plt.close(fig)
