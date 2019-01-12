class SequenceDataset(SingleDataset):
    # extended by folderUnlabelData
    
    def __init__(self, name, img_size, data_aug, maxscale, mean, std, seq_length):
        super(SequenceDataset, self).__init__(
            name, img_size, data_aug, 0, mean, std)
        self.seq_length = seq_length
        self.img_seqs = []
        self.episodes = []

        self.load_image_sequences()

        # total length
        self.N = 0
        for sequence in self.img_seqs:
            self.N += len(sequence) - seq_length + 1
            self.episodes.append(self.N)

    def read_debug(self):
        imgsN = sum([len(sequence) for sequence in self.img_seqs])

        print '{}: {} episodes, {} sequences, {} images'.format(self, len(self.img_seqs), self.N, imgsN)

    def load_image_sequences(self):
        # for Duke and VIRAT
        pass

    def save_sequence(self, sequence):
        # add new sequence to list if long enough
        if len(sequence) >= self.seq_length:
            # print 'sequence: ', len(sequence)

            self.img_seqs.append(sequence)
            sequence = []
        # else:
            # print '!sequence too short'

        return sequence

    def get_indexes(self, idx):
        ep_idx = 0  # calculate the episode index
        while idx >= self.episodes[ep_idx]:
            ep_idx += 1

        if ep_idx > 0:
            idx -= self.episodes[ep_idx - 1]

        return ep_idx, idx


class SingleSequenceDataset(SequenceDataset):
    # extended by DukeSequenceDataset, ViratSequenceDataset

    def __init__(self, name, img_size, data_aug, maxscale, mean, std, seq_length):
        super(SingleSequenceDataset, self).__init__(
            name, img_size, data_aug, maxscale, mean, std, seq_length)

    def __getitem__(self, idx):
        ep_idx, idx = self.get_indexes(idx)

        # random fliping
        flipping = self.get_flipping()

        imgseq = []
        labelseq = []
        for k in range(self.seq_length):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k][0])

            angle = self.img_seqs[ep_idx][idx + k][1]
            label = label_from_angle(angle)

            out_img, label = self.get_img_and_label(img, label, flipping)

            imgseq.append(out_img)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq': np.array(labelseq)}
