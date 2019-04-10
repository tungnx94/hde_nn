from utils import get_path, seq_show
import torch.optim as optim

from network import *
from dataset import *

def test_mobile_reg():
    net = MobileReg()
    net.load_mobilenet('network/pretrained_models/mobilenet_v1_0.50_224.pth')

    dataset = SingleLabelDataset(
        "duke-test", path=get_path('DukeMTMC/train.csv'), data_aug=True)
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=16)
    
    unlabel_set = SequenceUnlabelDataset('duke-unlabel', path=get_path('DukeMTMC/test_unlabel.csv'))
    unlabel_loader = DataLoader(unlabel_set) 

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for ind in range(1, 50): # 5000
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        unlabel_seq = unlabel_loader.next_sample().squeeze()

        # l = net.loss_label(imgseq, labels, mean=True)
        l = net.loss_combine(imgseq, labels, unlabel_seq, mean=True)
        print(l[0].item(), l[1].item(), l[2].item())

        optimizer.zero_grad()
        l[0].backward()
        optimizer.step()
        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())

def test_hde_rnn():
    net = HDE_RNN(extractor="base")

    dataset = DukeSeqLabelDataset('duke-seq', path=get_path("DukeMTMC/train.csv"), seq_length=8)
    loader = DataLoader(dataset, batch_size=1)

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for ind in range(1, 10):
        sample = loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        outputs = net(inputs)

        l = net.loss_label(inputs, targets, mean=True)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        seq_show(inputs.numpy(), dir_seq=outputs.to("cpu").detach().numpy())

def test_mobile_rnn():
    dataset = DukeSeqLabelDataset(
        "duke", path=get_path('DukeMTMC/train.csv'), seq_length=16, data_aug=True)
    dataset.shuffle()
    # dataset.resize(5000)
    loader = DataLoader(dataset)

    model = MobileRNN(rnn_type="gru", n_layer=2, rnnHidNum=128)

    optimizer = optim.Adam(model.parameters(), lr=0.0075)
    # train
    for ind in range(1, 20):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()
        
        #loss = model.forward_label(imgseq, labels)
        loss_w = model.loss_weighted(imgseq, labels, mean=True)
        loss = model.loss_label(imgseq, labels, mean=True)
        print(loss_w.item() , ' ', loss.item())

        optimizer.zero_grad()
        loss_w.backward()
        optimizer.step()

    print("Finished")

def test_hde_reg()
    net = HDEReg()
    dataset = SingleLabelDataset(
        "duke", path=get_path('DukeMTMC/test.csv'), img_size=192)
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=16)

    optimizer = optim.Adam(net.parameters(), lr=0.03)
    for ind in range(1, 50):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        loss = net.loss_label(imgseq, labels, mean=True)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_single_data():
    duke = SingleLabelDataset("duke", path=get_path('DukeMTMC/train.csv'), data_aug=True)
    pes = SingleLabelDataset("3dpes", path=get_path('3DPES/train.csv'), data_aug=True)
    # virat = SingleLabelDataset("virat", path=get_path('VIRAT/person/train.csv'), data_aug=True)

    for dataset in [duke, pes]:
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=32)
        for count in range(3):
            img, label = dataloader.next_sample()
            seq_show(img.numpy(),
                     dir_seq=label.numpy(), scale=0.5)

def test_duke_seq_data():
    unlabelset = DukeSeqLabelDataset("duke-test", path=get_path('DukeMTMC/test.csv'))
    dataloader = DataLoader(unlabelset)

    for count in range(5):
        sample = dataloader.next_sample()
        imgseq = sample[0].squeeze()
        labelseq = sample[1].squeeze()

        seq_show(imgseq.numpy(), dir_seq=labelseq)

def test_seq_unlabel_data():
    duke = SequenceUnlabelDataset('duke-unlabel', path=get_path('DukeMTMC/train_unlabel.csv'))
    ucf = SequenceUnlabelDataset('ucf-unlabel', path=get_path('UCF/train.csv'))
    drone = SequenceUnlabelDataset('drone-unlabel', path=get_path('DRONE_seq/train.csv'))

    for dataset in [duke, ucf, drone]:
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=1)
        for count in range(3):
            sample = dataloader.next_sample()
            seq_show(sample.squeeze().numpy(), scale=0.8)

def test_virat_seq_data():
    virat = ViratSeqLabelDataset("virat", path=get_path('VIRAT/person/train.csv'), seq_length=12)
    pes = ViratSeqLabelDataset("3dpes", path=get_path('3DPES/train.csv'), seq_length=12, data_aug=True)

    for dataset in [virat, pes]:
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=1)

        for count in range(3):
            imgseq, angleseq = dataloader.next_sample()
            imgseq = imgseq.squeeze().numpy()
            angleseq = angleseq.squeeze().numpy()

            seq_show(imgseq, dir_seq=angleseq, scale=0.8)

if __name__ == '__main__':
    # test_mobile_reg()
    # test_hde_rnn()
    # test_mobile_rnn()
    # test_hde_reg()
    # test_single_data()
    # test_duke_seq_data()
    # test_virat_seq_data()
    # test_seq_unlabel_data()