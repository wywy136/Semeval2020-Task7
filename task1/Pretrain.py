import Data
import Model
import Config
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import torch.nn.functional as F
import torch


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


torch.cuda.set_device(1)

dataset_pretrain = Data.HeadlineforPretraining()
dataset_pretrain.build()
batch_num = dataset_pretrain.get_batch_num(Config.args.pretrain_batch_size)

model = Model.BertPolarityPretrain()
model = model.to(Config.args.device)

optimizer = AdamW(model.parameters(), lr=Config.args.pretrain_rate, weight_decay=0.01)
training_steps = Config.args.pretrain_epoch_num * batch_num
warmup_steps = int(training_steps * Config.args.pretrain_warm_up)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

logger = get_logger(Config.args.pretrain_log_path)
for epoch in range(Config.args.pretrain_epoch_num):
    batch_generator = Data.generate_batches(dataset=dataset_pretrain, batch_size=Config.args.pretrain_batch_size)
    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        logits = model(batch_dict['input'].long(),
                       batch_dict['mask'].float(),
                       batch_dict['type'].long())
        loss = F.cross_entropy(logits, batch_dict['label'].long(), reduction='sum')
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_index % 100 == 0:
            logger.info('Epoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {}'.format(
                epoch, Config.args.pretrain_epoch_num, batch_index, batch_num, loss.item()
            ))

model.save()