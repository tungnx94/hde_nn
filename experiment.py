def conv2d():
    from utils import get_path
    from network import StateCoder 
    from dataset import DataLoader, FolderUnlabelDataset
    import torch.nn as nn

    hiddens = [3, 16, 32, 32, 64, 64, 128, 256]
    kernels = [4, 4, 4, 4, 4, 4, 3]
    paddings = [1, 1, 1, 1, 1, 1, 0]
    strides = [2, 2, 2, 2, 2, 2, 1]

    model = StateCoder(
        hiddens, kernels, strides, paddings, actfunc='leaky')
    dataset = FolderUnlabelDataset("ucf-unlabel", img_dir=get_path("UCF"))
    dataloader = DataLoader(dataset)  # batch_size = 1

    print model

    sample = dataloader.next_sample()

    x = sample
    print x.shape

    x = model.new_variable(sample.squeeze())
    print x.shape
    print ""

    for module in model.coder.modules():
        if isinstance(module, nn.Sequential) or isinstance(module, nn.LeakyReLU):
            continue 

        x = module(x)
        print module
        print x.shape
        print ""

def encoderReg():
    from network import MobileEncoderReg

    model = MobileEncoderReg()
    print model
    print "done"

def main():
    conv2d()
    # encoderReg()

if __name__ == '__main__':
    main()