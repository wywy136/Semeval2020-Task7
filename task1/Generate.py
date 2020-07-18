import torch
import Model
import Data
import Config
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(Config.args.cuda_index)

args = Config.args

model = Model.BERTModel_two_large()
model = model.to(args.device)
model.load_state_dict(torch.load("/home/ramon/chenshaowei_summer/wy/IC2/model/14_2.pth"))

test_dataset = Data.HeadlineDataset_two('test')
batch_generator = Data.generate_batches(dataset=test_dataset, batch_size=1, device=args.device)
num = test_dataset.get_batch_num(1)
print(num)

f = open('task-1-output-2.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(['id', 'pred'])

for batch_index, batch_dict in enumerate(batch_generator):
    # print(batch_index)
    pred = model(batch_dict['source'].long(),
                 batch_dict['mask'].float(),
                 batch_dict['type'].long())
    ans = pred.squeeze(1).float().item()
    csv_writer.writerow([str(batch_dict['idx'].item()), str(ans)])