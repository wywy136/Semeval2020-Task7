from transformers import BertForSequenceClassification
import transformers
import Config
import torch
import re
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv


torch.cuda.set_device(0)


def get_sentence_pair(sent_orig, edit_word):
    sent_o = re.sub("[</>]", "", sent_orig)
    sent_e = (sent_orig.split("<"))[0] + edit_word + (sent_orig.split(">"))[1]

    return sent_e, sent_o


class two_sentence_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.df = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        self.cls = [self.tokenizer.cls_token_id]
        self.sep = [self.tokenizer.sep_token_id]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        grade = torch.tensor(self.df["meanGrade"][idx])

        sent_e, sent_o = get_sentence_pair(
            self.df["original"][idx], self.df["edit"][idx]
        )

        sent_e_tokens = (
            self.cls
            + self.tokenizer.encode(sent_e, add_special_tokens=False)
            + self.sep
        )
        sent_o_tokens = (
            self.tokenizer.encode(sent_o, add_special_tokens=False) + self.sep
        )

        token_type_ids = (
            torch.tensor(
                [0] * len(sent_e_tokens) + [1] * (self.max_len - len(sent_e_tokens))
            )
        ).long()
        attention_mask = torch.tensor(
            [1] * (len(sent_o_tokens) + len(sent_e_tokens))
            + [0] * (self.max_len - len(sent_o_tokens) - len(sent_e_tokens))
        )
        attention_mask = attention_mask.float()

        input_ids = torch.tensor(sent_o_tokens + sent_e_tokens)

        if len(input_ids) < self.max_len:
            input_ids = torch.cat(
                (
                    input_ids,
                    (torch.ones(self.max_len - len(input_ids)) * self.pad).long(),
                )
            )
            token_type_ids = torch.cat(
                (
                    token_type_ids,
                    (torch.ones(self.max_len - len(token_type_ids)) * self.pad).long(),
                )
            )
        elif len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]

        id = self.df['id'][idx]
        id1 = int(id.split('-')[0])
        id2 = int(id.split('-')[1])

        return input_ids, token_type_ids, attention_mask, grade, torch.tensor(id1), torch.tensor(id2)


train_df = pd.read_csv("/home/ramon/chenshaowei_summer/wy/IC_final/data/task2/train.csv")
valid_df = pd.read_csv("/home/ramon/chenshaowei_summer/wy/IC_final/data/task2/dev.csv")
test_df = pd.read_csv("/home/ramon/chenshaowei_summer/wy/IC_final/data/task2/test.csv")

# x = train_df.append(valid_df, ignore_index=True)
# total_data_df = x.append(test_df, ignore_index=True)
total_data_df = test_df

model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

total_df_1 = total_data_df[["id", "original1", "edit1", "meanGrade1"]]
total_df_2 = total_data_df[["id", "original2", "edit2", "meanGrade2"]]
total_df_1.rename(
    columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"},
    inplace=True,
)
total_df_2.rename(
    columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"},
    inplace=True,
)

total_dset_1 = two_sentence_dataset(total_df_1, tokenizer, max_len=256)
total_dset_2 = two_sentence_dataset(total_df_2, tokenizer, max_len=256)

total_loader_1 = DataLoader(total_dset_1, batch_size=32, shuffle=False, num_workers=0)
total_loader_2 = DataLoader(total_dset_2, batch_size=32, shuffle=False, num_workers=0)

# if LOAD_LM:
#     fname = "without_LM" + model_name + "_task1"
# else:
#     fname = model_name + "_task1"

# PREDICTIONS ON TEST

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=1
)
model = model.to(Config.args.device)
model.load_state_dict(torch.load("/home/ramon/chenshaowei_summer/wy/IC_final/task1/model/1_2.pt"))
model.eval()

preds_1 = []
idx = []
with torch.no_grad():
    for i, data in enumerate(total_loader_1):
        input_ids, token_type_ids, attention_mask, grade, g1, g2 = data
        input_ids, token_type_ids, attention_mask, grade = (
            input_ids.to(Config.args.device),
            token_type_ids.to(Config.args.device),
            attention_mask.to(Config.args.device),
            grade.to(Config.args.device),
        )
        outputs = model(
            input_ids=input_ids,
            labels=grade,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _, logits = outputs[:2]

        for logit in logits.reshape(-1):
            preds_1.append(logit.item())
        for k in range(input_ids.size(0)):
            idx.append((g1[k].item(), g2[k].item()))

f = open('task-2-output.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(['id', 'pred'])

preds_2 = []
with torch.no_grad():
    for i, data in enumerate(total_loader_2):
        input_ids, token_type_ids, attention_mask, grade, g1, g2 = data
        input_ids, token_type_ids, attention_mask, grade = (
            input_ids.to(Config.args.device),
            token_type_ids.to(Config.args.device),
            attention_mask.to(Config.args.device),
            grade.to(Config.args.device),
        )
        outputs = model(
            input_ids=input_ids,
            labels=grade,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _, logits = outputs[:2]

        for logit in logits.reshape(-1):
            preds_2.append(logit.item())

for i in range(len(preds_1)):
    ans = 0
    if preds_1[i] > preds_2[i]:
        ans = 1
    elif preds_1[i] < preds_2[i]:
        ans = 2
    elif preds_1[i] == preds_2[i]:
        ans = 3
    csv_writer.writerow(["{}-{}".format(idx[i][0], idx[i][1]), str(ans)])