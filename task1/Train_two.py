import Data
import Model
import Config
import torch
from torch.nn import functional as F
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import logging
import os


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


args = Config.args
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda_index)
torch.cuda.set_device(1)
seed_num = 1234
torch.manual_seed(seed_num)

train_dataset = Data.HeadlineDataset_oe('train')
dev_dataset = Data.HeadlineDataset_oe('dev')
test_dataset = Data.HeadlineDataset_oe('test')
batch_num_train = train_dataset.get_batch_num(args.batch_size)
batch_num_dev = dev_dataset.get_batch_num(1)
batch_num_test = dev_dataset.get_batch_num(1)

# model = Model.BERTModel_two()
# model = model.to(args.device)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model = model.to(args.device)
# param_optimizer = list(model.named_parameters())
# optimizer_grouped_parameters = [
#                 {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': args.weight_decay},
#                 {'params': [p for n, p in param_optimizer if "_bert" not in n],
#                  'lr': args.learning_rate, 'weight_decay': args.weight_decay}]
# optimizer = AdamW(optimizer_grouped_parameters, lr=args.finetuning_rate, correct_bias=False)
for name, param in model.named_parameters():
    if (
        ("_linear" not in name)
        & ("pooler" not in name)
        & ("layer.11" not in name)
        & ("layer.10" not in name)
    ):
        param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=10e-8)

# training_steps = args.epoch_num * batch_num_train
# warmup_steps = int(training_steps * args.warm_up)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
logger = get_logger(args.log_path)

min_dev_mse = 10.
min_test_mse = 10.
MSEloss = torch.nn.MSELoss(reduction='sum')
logger.info('Batch size: {}\tLearning Rate: {}\tFinetuning Rate: {}\t'.format(args.batch_size,
                                                                              args.learning_rate,
                                                                              args.finetuning_rate))
logger.info('Num layer: {}\tHidden Size: {}\tDropout: {}\t'.format(args.num_layers,
                                                                   args.rnn_hidden_size,
                                                                   args.dropout))

for epoch in range(args.epoch_num):

    model.train()
    batch_generator = Data.generate_batches(dataset=train_dataset, batch_size=args.batch_size, device=args.device)

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        loss, pred = model(batch_dict['source'].long(),
                           batch_dict['mask'].float(),
                           batch_dict['type'].long(),
                           labels=batch_dict['meangrade'].float())[:2]
        # loss = MSEloss(pred.squeeze(1).float(), batch_dict['meangrade'].float())

        loss.backward()
        optimizer.step()
        # scheduler.step()
        # model.zero_grad()

        if batch_index % 50 == 0:
            logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t'.format(epoch, args.epoch_num,
                                                                               batch_index, batch_num_train,
                                                                               round(loss.item(), 6)))
    # logger.info('Validating...:')
    model.eval()
    batch_generator = Data.generate_batches(dataset=dev_dataset, batch_size=1, device=args.device)

    MSE_sum = 0.
    with torch.no_grad():
        for batch_index, batch_dict in enumerate(batch_generator):
            loss, pred = model(batch_dict['source'].long(),
                               batch_dict['mask'].float(),
                               batch_dict['type'].long(),
                               labels=batch_dict['meangrade'].float())[:2]
            # loss = MSEloss(pred.squeeze(1).float(), batch_dict['meangrade'].float())
            # print(batch_index)
            MSE_sum += loss.item()

    MSE_sum = MSE_sum / batch_num_dev
    # print(batch_num_dev)
    logger.info('Validation MSE: {}'.format(MSE_sum))
    if MSE_sum < min_dev_mse:
        min_dev_mse = MSE_sum
        logger.info('Model saved after epoch {}'.format(epoch))
        torch.save(model.state_dict(), args.model_path + '{}.pth'.format(epoch))

    batch_generator = Data.generate_batches(dataset=test_dataset, batch_size=1, device=args.device)
    MSE_sum_test = 0.
    with torch.no_grad():
        for batch_index, batch_dict in enumerate(batch_generator):
            loss, pred = model(batch_dict['source'].long(),
                         batch_dict['mask'].float(),
                         batch_dict['type'].long(),
                               labels=batch_dict['meangrade'].float())[:2]
            # loss = MSEloss(pred.squeeze(1).float(), batch_dict['meangrade'].float())
            # print(batch_index)
            MSE_sum_test += loss.item()

    MSE_sum_test = MSE_sum_test / batch_num_test
    logger.info('Test MSE: {}'.format(MSE_sum_test))
    if MSE_sum_test < min_test_mse:
        min_test_mse = MSE_sum_test
        logger.info('Model saved after epoch {}'.format(epoch))
        torch.save(model.state_dict(), args.model_path + '{}.pth'.format(epoch))