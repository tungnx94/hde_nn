import torch.optim as optim

import cv2
from utils import *
from network import *
from dataset import *


def test_hde_reg():
    config = read_json("config/train0a.json")
    net = HDEReg(config["model"])
    dataset = SingleLabelDataset(config["dataset"]["train"])
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=16)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for ind in range(1, 20):
        sample = loader.next_sample()
        imgseq, labels = sample

        loss = net.loss_label(imgseq, labels, mean=True)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_mobile_reg():
    config = read_json("config/train2a.json")
    net = MobileReg(config["model"])

    dataset = SingleLabelDataset(config["dataset"]["train"])
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=4)

    unlabel_set = SequenceUnlabelDataset(config["dataset"]["unlabel"])
    unlabel_loader = DataLoader(unlabel_set)

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for ind in range(1, 20):  # 5000
        sample = loader.next_sample()
        unlabel_seq = unlabel_loader.next_sample().squeeze()

        imgseq, labels = sample

        # l = net.loss_label(imgseq, labels, mean=True)
        l = net.loss_combine(imgseq, labels, unlabel_seq, mean=True)
        print(l[0].item(), l[1].item(), l[2].item())

        optimizer.zero_grad()
        l[0].backward()
        optimizer.step()
        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())


def test_hde_rnn():
    config = read_json("config/train3a.json")
    net = HDE_RNN(config["model"])

    dataset = DukeSeqLabelDataset(config["dataset"]["train"])
    loader = DataLoader(dataset, batch_size=8)

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for ind in range(1, 100):
        sample = loader.next_sample()
        inputs, targets = sample

        l = net.loss_label(inputs, targets, mean=True)
        print(l)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()


def test_mobile_rnn():
    config = read_json("config/train1a.json")

    dataset = DukeSeqLabelDataset(config["dataset"]["train"])
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=4)

    net = MobileRNN(config["model"])

    optimizer = optim.Adam(net.parameters(), lr=0.003)
    # train
    for ind in range(1, 20):
        sample = loader.next_sample()
        inputs, targets = sample

        l = net.loss_label(inputs, targets, mean=True)
        print(l)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print("Finished")

def test_feature_rnn():
    config = read_json("cnf_temp.json")
    net = FeatureRNN(config["model"])
    dataset = SingleLabelDataset(config["dataset"]["train"])
    loader = DataLoader(dataset, batch_size=4)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for ind in range(1, 20):
        sample = loader.next_sample()
        imgseq, labels = sample

        loss = net.loss_label(imgseq, labels, mean=True)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_single_data():
    config = read_json("config/data.json")["single"]
    duke = SingleLabelDataset(config["duke"])
    #pes = SingleLabelDataset(config["pes"])

    for dataset in [duke]:
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=10)
        for count in range(3):
            img, label = dataloader.next_sample()
            seq_show(img.numpy(),
                     dir_seq=label.numpy(), scale=0.5)


def test_duke_seq_data():
    config = read_json("config/data.json")["label_seq"]
    dts1 = DukeSeqLabelDataset(config["duke_s1"])
    dts2 = DukeSeqLabelDataset(config["duke_s2"])

    for dts in [dts1, dts2]:
        dataloader = DataLoader(dts)    
        sample = dataloader.next_sample()
        imgseq = sample[0].squeeze(dim=0)
        labelseq = sample[1].squeeze(dim=0)
        img = seq_show(imgseq.numpy())

        cv2.imwrite(dts.__str__() + ".jpg", img)


def test_seq_unlabel_data():
    config = read_json("config/data.json")["unlabel"]

    duke_tr = SequenceUnlabelDataset(config["duke1"])
    duke_v = SequenceUnlabelDataset(config["duke2"])
    duke_te = SequenceUnlabelDataset(config["duke3"])
    #ucf = SequenceUnlabelDataset(config["ucf"])
    #drone = SequenceUnlabelDataset(config["drone"])
    #dpes = ViratUnlabelDataset(config["3dpes"])

    #for dataset in [duke, ucf, drone]:
    for dataset in [duke_tr, duke_v, duke_te]:
        print(dataset)
        dataloader = DataLoader(dataset, batch_size=1)
        for count in range(3):
            sample = dataloader.next_sample()
            seq_show(sample.squeeze().numpy(), scale=0.8)


def calculate_duke_scales():
    config = read_json("config/data.json")["unlabel"]
    duke = UnlabelDataset(config["duke"])

    mean, std = duke.calculate_mean_std()
    d = {"mean": mean.tolist(), "std": std.tolist()}
    write_json(d, "config/duke2.json")


def calculate_ucf_scales():
    config = read_json("config/data.json")["unlabel"]
    ucf = UnlabelDataset(config["ucf"])

    mean, std = ucf.calculate_mean_std()
    d = {"mean": mean.tolist(), "std": std.tolist()}
    write_json(d, "config/ucf2.json")


def calculate_indoor_scales():
    config = read_json("config/data.json")["unlabel"]
    indoor = UnlabelDataset(config["indoor"])

    mean, std = indoor.calculate_mean_std()
    d = {"mean": mean.tolist(), "std": std.tolist()}
    write_json(d, "config/indoor.json")

if __name__ == '__main__':
    # test_hde_reg()
    # test_mobile_reg()

    # test_hde_rnn()
    # test_mobile_rnn()
    #test_feature_rnn()

    #test_single_data()
    test_duke_seq_data()
    #test_seq_unlabel_data()

    #calculate_duke_scales()
    #calculate_ucf_scales()
    # calculate_indor_scales()
