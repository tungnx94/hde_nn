import matplotlib.pyplot as plt

from .exception import WFException


class AccumulatedValuePlotter(object):

    def __init__(self, name, av, avNameList, plot_average=False):
        self.name = name
        self.AV = av
        self.avNameList = avNameList # variables to plot
        self.plot_average = plot_average 

        if len(self.avNameList) == 0:
            exp = WFException("The avNameList is empty.",
                              "AccumulatedValuePlotter")
            raise(exp)
            
    def save_plot(self, fig, ax, legend, filePath):
        # plot
        ax.legend(legend)
        ax.grid()
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")

        # save image & close
        fig.savefig(filePath)
        plt.close(fig)          

    def plot_update(self, ax):
        legend = []
        stamps = self.AV.get_stamps()
        for name in self.avNameList:
            ax.plot(stamps, self.AV.get_values(name)) 
            legend.append(name)

            if self.plot_average:
                ax.plot(stamps, self.AV.get_avg_values(name))
                legend.append(name + "_avg")

        return ax, legend 

    def plot_final(self, ax):
        legend = []
        stamps = self.AV.get_stamps()
        for name in self.avNameList:
            ax.plot(stamps, self.AV.get_values(name)) 
            legend.append(name.split("_")[0])
        return ax, legend 

    def plot_avg_final(self, ax):
        legend = []
        stamps = self.AV.get_stamps()
        for name in self.avNameList:
            ax.plot(stamps, self.AV.get_avg_values(name)) 
            legend.append(name.split("_")[0])
        return ax, legend 

    def write_image(self, outDir):
        """ write loss diagramm to image file """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax, legend = self.plot_update(ax)
    
        self.save_plot(fig, ax, legend, outDir + '/' + self.name + ".png")

    def write_image_final(self, outDir):
        # all values with large oscilation
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax, legend = self.plot_final(ax)
        self.save_plot(fig, ax, legend, outDir + '/' + self.name + "_final.png")

        # average values -> "smooth" lines
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax, legend = self.plot_avg_final(ax)
        self.save_plot(fig, ax, legend, outDir + '/' + self.name + "_avg_final.png")      

    def update(self):
        return

    def initialize(self):
        print("AVPlotter initialized")
