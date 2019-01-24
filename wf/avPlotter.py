import matplotlib.pyplot as plt

from exception import WFException


class AccumulatedValuePlotter(object):

    def __init__(self, name, av, avNameList, plot_average=True):
        self.name = name
        self.AV = av
        self.avNameList = avNameList
        self.plot_average = plot_average 

        if len(self.avNameList) == 0:
            exp = WFException("The avNameList is empty.",
                              "AccumulatedValuePlotter")
            raise(exp)
            
        self.lastIdx = -1

    def write_image(self, outDir):
        """ write loss diagramm to image file """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        legend = []

        stamps = self.AV.get_stamps()
        for name in self.avNameList:
            ax.plot(stamps, self.AV.get_values(name)) 
            legend.append(name)

            if self.plot_average:
                ax.plot(stamps, self.AV.get_avg_values(name))
                legend.append(name + "_avg")

        # plot
        ax.legend(legend)
        ax.grid()
        ax.set_title(self.name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")

        fig.savefig(outDir + "/" + self.title + ".png")
        plt.close(fig)

    def update(self):
        return

    def initialize(self):
        print "AVPlotter initialized"
