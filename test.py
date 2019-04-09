def test_mobile_reg():
    from utils import get_path, seq_show
    import torch.optim as optim
    from network import MobileReg
    from dataset import SingleLabelDataset, DukeSeqLabelDataset, SequenceUnlabelDataset, DataLoader

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

    print("Finished")

if __name__ == '__main__':
    test_mobile_reg()
