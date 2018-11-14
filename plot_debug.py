import os
import numpy as np
import matplotlib.pyplot as plt

def groupPlot(data_x, data_y, group=10):
    """ plot data by group, each using mean of coordinates """
    data_x, data_y = np.array(data_x), np.array(data_y)

    # truncate length
    d_len = len(data_x) / group * group
    data_x = data_x[0: d_len]
    data_y = data_y[0: d_len]

    data_x, data_y = data_x.reshape((-1, group)), data_y.reshape((-1, group))
    data_x, data_y = data_x.mean(axis=1), data_y.mean(axis=1)
    return (data_x, data_y)


def main():

    exp_ind = '1_1_'
    datadir = 'logdata'
    filelist = [['loss', 'test_loss'],
                ['label_loss', 'test_label'],
                ['unlabel_loss', 'test_unlabel'],
                ]

    labellist = [['training loss', 'validation loss'],
                 ['training loss', 'validation loss'],
                 ['training loss', 'validation loss'],
                 ]

    titlelist = ['loss',
                 'label',
                 'unlabel',
                 ]
    imgoutdir = 'resimg_facing'
    AvgNum = 100

    for ind, files in enumerate(filelist):
        print ind, files
        ax = plt.subplot(int('22' + str(ind + 1)))
        # lines = []

        for k, filename in enumerate(files):

            filename = exp_ind + filename + '.npy'
            loss = np.load(os.path.join(datadir, filename))
            print loss.shape
            if k == 1:  # test data
                loss[:, 0] = loss[:, 0] * 10
                datax, datay = groupPlot(loss[:, 0], loss[:, 1], group=1)
                ax.plot(datax, datay, label=labellist[ind][k])
                
            # ax.plot(loss[:,0],loss[:,1], label=labellist[ind][k])
            if k == 0:
                datax, datay = groupPlot(loss[:, 0], loss[:, 1], group=1)
                ax.plot(datax, datay, label=labellist[ind][k])

        ax.grid()
        ax.legend()
        # ax.set_ylim(0,0.8)
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('loss')
        ax.set_title(titlelist[ind])

    plt.show()

if __name__ == "__main__":
    main()
