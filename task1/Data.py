import torch
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import re
import nltk


class HeadlineDataset(Dataset):
    def __init__(self, type):
        super(HeadlineDataset, self).__init__()
        self.source = []
        self.mask = []
        self.grade = []
        self.meangrade = []
        self.max_length = 0

        self.t = BertTokenizer.from_pretrained('bert-base-uncased')
        if type == 'train':
            with open('../data/task1/train.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
                # print(self._data[62])
            # with open('./data/train_funlines.csv', 'r', encoding='utf-8') as f:
            #     self._data += list(csv.reader(f))[1:]
        elif type == 'dev':
            with open('../data/task1/dev.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
        elif type == 'test':
            with open('../data/task1/test.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]

        for d in self._data:
            if len(d) != 5:
                continue
            self.max_length = max(self.max_length, len(d[1].split(' ')))
        self.max_length += 2

        for d in self._data:
            if len(d) != 5:
                continue
            src = d[1].split(' ')
            for i in range(len(src)):
                if '/>' in src[i]:
                    src[i] = d[2]

            self.source.append(self.t.convert_tokens_to_ids([w.lower() for w in src]))
            self.source[-1].insert(0, self.t.convert_tokens_to_ids('[CLS]'))
            self.source[-1].append(self.t.convert_tokens_to_ids('[SEP]'))
            self.mask.append([1 for i in range(len(self.source[-1]))])

            self.source[-1].extend([0] * (self.max_length - len(self.source[-1])))
            self.mask[-1].extend([0] * (self.max_length - len(self.mask[-1])))

            self.meangrade.append(float(d[4]))
            self.grade.append(int(d[3]))

    def get_batch_num(self, batch_size):
        return len(self._data) // batch_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return {'source': np.array(self.source[item]),
                'grade': self.grade[item],
                'meangrade': self.meangrade[item],
                'mask': np.array(self.mask[item])}


class HeadlineDataset_two(Dataset):
    def __init__(self, type):
        super(HeadlineDataset_two, self).__init__()
        self.source = []
        self.mask = []
        self.grade = []
        self.meangrade = []
        self.max_length = 0
        self.type = []
        self.idx = []
        self.pos = []

        self.t = BertTokenizer.from_pretrained('bert-base-uncased')
        if type == 'train':
            with open('../data/task1/train.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
                # print(self._data[62])
            # with open('./data/train_funlines.csv', 'r', encoding='utf-8') as f:
            #     self._data += list(csv.reader(f))[1:]
        elif type == 'dev':
            with open('../data/task1/dev.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
        elif type == 'test':
            with open('../data/task1/test.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]

        for d in self._data:
            if len(d) != 5:
                continue
            self.max_length = max(self.max_length, len(d[1].split(' ')))
        self.max_length += 3

        for d in self._data:
            if len(d) != 5:
                continue
            src = d[1].split(' ')
            for i in range(len(src)):
                if '/>' in src[i]:
                    src[i] = d[2]
                    self.pos.append(i + 1)
                    break

            self.source.append(self.t.convert_tokens_to_ids([w.lower() for w in src]))
            self.source[-1].insert(0, self.t.convert_tokens_to_ids('[CLS]'))
            self.source[-1].append(self.t.convert_tokens_to_ids('[SEP]'))
            self.type.append([0 for i in range(len(self.source[-1]))])

            self.source[-1].append(self.t.convert_tokens_to_ids(d[2]))
            self.mask.append([1 for i in range(len(self.source[-1]))])

            self.source[-1].extend([0] * (self.max_length - len(self.source[-1])))
            self.mask[-1].extend([0] * (self.max_length - len(self.mask[-1])))
            self.type[-1].extend([1] * (self.max_length - len(self.type[-1])))

            self.meangrade.append(float(d[4]))
            self.grade.append(int(d[3]))

            self.idx.append(int(d[0]))

    def get_batch_num(self, batch_size):
        return len(self._data) // batch_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return {'source': np.array(self.source[item]),
                'grade': self.grade[item],
                'meangrade': self.meangrade[item],
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'idx': self.idx[item],
                'pos': self.pos[item]
                }


class HeadlineDataset_enhanced(Dataset):
    def __init__(self, type):
        super(HeadlineDataset_enhanced, self).__init__()
        self.source = []
        self.mask = []
        self.grade = []
        self.meangrade = []
        self.max_length = 0
        self.type = []
        self.idx = []

        self.t = BertTokenizer.from_pretrained('bert-base-uncased')
        if type == 'train':
            with open('./data/train.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
                # print(self._data[62])
            with open('./data/train_funlines.csv', 'r', encoding='utf-8') as f:
                self._data += list(csv.reader(f))[1:]
        elif type == 'dev':
            with open('./data/dev.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
        elif type == 'test':
            with open('./data/test.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]

        enhance_data = []
        for d in self._data:
            if len(d) != 5:
                continue
            self.max_length = max(self.max_length, len(d[1].split(' ')))
            if float(d[4]) >= 2:
                for j in range(1):
                    enhance_data.append(d)
        self.max_length += 3
        self._data += enhance_data

        for d in self._data:
            if len(d) != 5:
                continue
            src = d[1].split(' ')
            for i in range(len(src)):
                if '/>' in src[i]:
                    src[i] = d[2]

            self.source.append(self.t.convert_tokens_to_ids([w.lower() for w in src]))
            self.source[-1].insert(0, self.t.convert_tokens_to_ids('[CLS]'))
            self.source[-1].append(self.t.convert_tokens_to_ids('[SEP]'))
            self.type.append([0 for i in range(len(self.source[-1]))])

            self.source[-1].append(self.t.convert_tokens_to_ids(d[2]))
            self.mask.append([1 for i in range(len(self.source[-1]))])

            self.source[-1].extend([0] * (self.max_length - len(self.source[-1])))
            self.mask[-1].extend([0] * (self.max_length - len(self.mask[-1])))
            self.type[-1].extend([1] * (self.max_length - len(self.type[-1])))

            self.meangrade.append(float(d[4]))
            self.grade.append(int(d[3]))

            self.idx.append(int(d[0]))


    def get_batch_num(self, batch_size):
        return len(self._data) // batch_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return {'source': np.array(self.source[item]),
                'grade': self.grade[item],
                'meangrade': self.meangrade[item],
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'idx': self.idx[item]
                }


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cuda"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            out_dict[name] = data_dict[name].to(device)
        yield out_dict


class HeadlineDataset_oe(Dataset):
    def __init__(self, type):
        super(HeadlineDataset_oe, self).__init__()
        self.source = []
        self.mask = []
        self.grade = []
        self.meangrade = []
        self.max_length = 0
        self.type = []
        self.idx = []
        self.pos = []

        self.t = BertTokenizer.from_pretrained('bert-base-uncased')
        if type == 'train':
            with open('../data/task1/train.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
                # print(self._data[62])
            # with open('./data/train_funlines.csv', 'r', encoding='utf-8') as f:
            #     self._data += list(csv.reader(f))[1:]
        elif type == 'dev':
            with open('../data/task1/dev.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]
        elif type == 'test':
            with open('../data/task1/test.csv', 'r', encoding='utf-8') as f:
                self._data = list(csv.reader(f))[1:]

        for d in self._data:
            if len(d) != 5:
                continue
            self.max_length = max(self.max_length, len(d[1].split(' ')))
        self.max_length += self.max_length + 3

        for d in self._data:
            if len(d) != 5:
                continue
            src = d[1]
            sent_o = re.sub("[</>]", "", src)
            sent_e = src.split("<")[0] + d[2] + src.split(">")[1]

            self.source.append(self.t.convert_tokens_to_ids([w.lower() for w in sent_o.split(' ')]))
            self.source[-1].insert(0, self.t.convert_tokens_to_ids('[CLS]'))
            self.source[-1].append(self.t.convert_tokens_to_ids('[SEP]'))
            self.type.append([0 for i in range(len(self.source[-1]))])

            self.source[-1] += self.t.convert_tokens_to_ids([w.lower() for w in sent_e.split(' ')])
            self.source[-1].append(self.t.convert_tokens_to_ids('[SEP]'))
            self.mask.append([1 for i in range(len(self.source[-1]))])

            self.source[-1].extend([0] * (self.max_length - len(self.source[-1])))
            self.mask[-1].extend([0] * (self.max_length - len(self.mask[-1])))
            self.type[-1].extend([1] * (self.max_length - len(self.type[-1])))

            self.meangrade.append(float(d[4]))
            self.grade.append(int(d[3]))

            self.idx.append(int(d[0]))

    def get_batch_num(self, batch_size):
        return len(self._data) // batch_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return {'source': np.array(self.source[item]),
                'grade': self.grade[item],
                'meangrade': self.meangrade[item],
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'idx': self.idx[item]
                }


class HeadlineforPretraining(Dataset):
    def __init__(self):
        super(HeadlineforPretraining,self).__init__()
        self.data_path = "/home/ramon/chenshaowei_summer/wy/IC_final/data/dataforpretrain/dataset.csv"
        self.input = []
        self.label = []
        self.mask = []
        self.type = []
        self.t = BertTokenizer.from_pretrained('bert-base-uncased')

    def build(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = list(csv.reader(f))[1:]

        max_length = 0
        for d in data:
            self.input.append([self.t.convert_tokens_to_ids('[CLS]')])
            self.input[-1] += self.t.convert_tokens_to_ids([w.lower() for w in nltk.word_tokenize(d[0])])
            self.input[-1] += [self.t.convert_tokens_to_ids('[SEP]')]
            max_length = max(max_length, len(self.input[-1]))

            self.mask.append([1 for i in range(len(self.input[-1]))])
            self.type.append([0 for i in range(len(self.input[-1]))])

            self.label.append(0 if d[1] == 'False' else 1)

        for i in range(len(self.input)):
            self.input[i].extend([0] * (max_length - len(self.input[i])))
            self.mask[i].extend([0] * (max_length - len(self.mask[i])))
            self.type[i].extend([1] * (max_length - len(self.type[i])))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return {'input': np.array(self.input[item]),
                'mask': np.array(self.mask[item]),
                'type': np.array(self.type[item]),
                'label': self.label[item]}

    def get_batch_num(self, bs):
        return len(self.input) // bs