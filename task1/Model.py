from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import Config


class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._linear = nn.Linear(768, 1)
        self._out = nn.Linear(Config.args.hidden_size, 1)

    def forward(self, source, mask):
        hidden = self._bert(source, attention_mask=mask)[0]
        # hidden = self._linear(hidden[:, 0, :])
        return self._linear(hidden[:, 0, :])


class BERTModel_two(nn.Module):
    def __init__(self):
        super(BERTModel_two, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._linear = nn.Linear(768, 1)
        # self._out = nn.Linear(Config.args.hidden_size, 1)

    def forward(self, source, mask, type):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        # hidden = self._linear(hidden[:, 0, :])
        return self._linear(hidden[:, 0, :])


class BERTModel_two_large(nn.Module):
    def __init__(self):
        super(BERTModel_two_large, self).__init__()
        self._bert = BertModel.from_pretrained('bert-large-uncased')
        self._linear = nn.Linear(1024, 1)
        self._out = nn.Linear(Config.args.hidden_size, 1)

    def forward(self, source, mask, type):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        # hidden = self._linear(hidden[:, 0, :])
        return self._linear(hidden[:, 0, :])


class BERTModel_conv(nn.Module):
    def __init__(self):
        super(BERTModel_conv, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._conv = nn.Conv2d(1, 32, (2, 768), 1)
        self._linear = nn.Linear(928, 1)

    def forward(self, source, mask, type):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        hidden = self._conv(hidden.unsqueeze(1))
        hidden = hidden.view(hidden.size(0), -1)
        return self._linear(hidden)


class BERTModel_concat(nn.Module):
    def __init__(self):
        super(BERTModel_concat, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._linear = nn.Linear(768 * 2, 1)
        self._out = nn.Linear(Config.args.hidden_size, 1)

    def forward(self, source, mask, type, pos, status):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        if status == 'train':
            new_hidden = torch.zeros([Config.args.batch_size, 768 * 2], dtype=torch.float).to(Config.args.device)
            for i in range(Config.args.batch_size):
                new_hidden[0] = torch.cat((hidden[i, 0, :].float(), hidden[i, pos[i].item(), :].float()))
        else:
            new_hidden = torch.cat((hidden[0, 0, :].float(), hidden[0, pos[0].item(), :].float())).to(Config.args.device)

        return self._linear(new_hidden)


class BERTModel_GRU(nn.Module):
    def __init__(self):
        super(BERTModel_GRU, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._gru = nn.GRU(768, Config.args.rnn_hidden_size,
                           batch_first=True,
                           bidirectional=Config.args.bidirectional,
                           num_layers=Config.args.num_layers)
        if Config.args.bidirectional:
            linear_hidden_in = Config.args.rnn_hidden_size * 2
        else:
            linear_hidden_in = Config.args.rnn_hidden_size
        self._linear = nn.Linear(linear_hidden_in, 1)
        self.dropout = nn.Dropout(Config.args.dropout)
        # self._out = nn.Linear(Config.args.hidden_size, 1)

    def forward(self, source, mask):
        hidden = self._bert(source, attention_mask=mask)[0]  # [B, L, 768]
        all_hidden, last_hidden = self._gru(hidden)
        # last_hidden = [num_layers * num_directions, B, R_H]
        if Config.args.bidirectional:
            last_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=-1)
        else:
            last_hidden = last_hidden.squeeze(0)
        return self._linear(self.dropout(last_hidden))


class BERTModel_two_GRU(nn.Module):
    def __init__(self):
        super(BERTModel_two_GRU, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._gru = nn.GRU(768, Config.args.rnn_hidden_size,
                           batch_first=True,
                           bidirectional=Config.args.bidirectional,
                           num_layers=Config.args.num_layers)
        if Config.args.bidirectional:
            linear_hidden_in = Config.args.rnn_hidden_size * 2
        else:
            linear_hidden_in = Config.args.rnn_hidden_size
        self._linear = nn.Linear(linear_hidden_in, 1)
        self.dropout = nn.Dropout(Config.args.dropout)

    def forward(self, source, mask, type):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        all_hidden, last_hidden = self._gru(hidden)
        # last_hidden = [num_layers * num_directions, B, R_H]
        if Config.args.bidirectional:
            last_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=-1)
        else:
            last_hidden = last_hidden.squeeze(0)
        return self._linear(self.dropout(last_hidden))


class BERTModel_two_LSTM(nn.Module):
    def __init__(self):
        super(BERTModel_two_LSTM, self).__init__()
        self._bert = BertModel.from_pretrained('bert-base-uncased')
        self._lstm = nn.LSTM(768, Config.args.rnn_hidden_size,
                            batch_first=True,
                            bidirectional=Config.args.bidirectional,
                            num_layers=Config.args.num_layers)
        if Config.args.bidirectional:
            linear_hidden_in = Config.args.rnn_hidden_size * 2
        else:
            linear_hidden_in = Config.args.rnn_hidden_size
        self._linear = nn.Linear(linear_hidden_in, 1)
        self.dropout = nn.Dropout(Config.args.dropout)

    def forward(self, source, mask, type):
        hidden = self._bert(source, attention_mask=mask, token_type_ids=type)[0]
        all_hidden, (last_hidden, last_c) = self._lstm(hidden)
        # last_hidden = [num_layers * num_directions, B, R_H]
        if Config.args.bidirectional:
            last_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=-1)
        else:
            last_hidden = last_hidden.squeeze(0)
        return self._linear(self.dropout(last_hidden))


class BertPolarityPretrain(nn.Module):
    def __init__(self):
        super(BertPolarityPretrain, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 2)

    def forward(self, input, mask, type):
        return self.linear(self.bert(input, mask, type)[0][:, 0, :])

    def save(self):
        self.bert.save_pretrained(Config.args.pretrained_model_path)