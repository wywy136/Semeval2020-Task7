from argparse import Namespace
import torch


args = Namespace(device=torch.device('cuda'),
                 cuda_index=1,
                 hidden_size=64,
                 epoch_num=10,
                 pretrain_epoch_num=10,
                 batch_size=32,
                 pretrain_batch_size=64,
                 learning_rate=4e-5,
                 finetuning_rate=2e-5,
                 pretrain_rate=1e-7,
                 warm_up=0.1,
                 pretrain_warm_up=0.1,
                 weight_decay=0.1,
                 model_path='./model/4_',
                 log_path='./log/4.log',
                 pretrain_log_path='./pretrainlog/1.log',
                 pretrained_model_path='./pretrained_model/',
                 bidirectional=True,
                 rnn_hidden_size=64,
                 dropout=0.2,
                 num_layers=1)