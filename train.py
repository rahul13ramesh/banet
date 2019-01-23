# coding: utf-8

from __future__ import print_function
from builtins import range
import os
import sys
import shutil
import pickle
from utils import blockPrint, enablePrint
from caption import Vocabulary
from data import get_train_loader
from model import BANet
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from evaluate import evaluate
from args import vocab_pkl_path, train_caption_pkl_path, feature_h5_path
from args import num_epochs, batch_size, learning_rate, ss_factor
from args import projected_size, hidden_size, mid_size
from args import feature_size, max_frames, max_words
from args import use_cuda, use_checkpoint
from args import banet_pth_path, optimizer_pth_path
from args import best_banet_pth_path, best_optimizer_pth_path
from args import test_range, test_prediction_txt_path, test_reference_txt_path
#  from args import log_environment
#  from tensorboard_logger import configure, log_value
sys.path.append('./coco-caption/')
from pycocotools.coco import COCO

#  configure(log_environment, flush_secs=10)

# 加载词典
with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 构建模型
banet = BANet(feature_size, projected_size, mid_size, hidden_size,
              max_frames, max_words, vocab)


if os.path.exists(banet_pth_path) and use_checkpoint:
    banet.load_state_dict(torch.load(banet_pth_path))
if use_cuda:
    banet.cuda()

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(banet.parameters(), lr=learning_rate)
if os.path.exists(optimizer_pth_path) and use_checkpoint:
    optimizer.load_state_dict(torch.load(optimizer_pth_path))

# 打印训练环境的参数设置情况
print('Learning rate: %.4f' % learning_rate)
print('Batch size: %d' % batch_size)

# 初始化数据加载器
train_loader = get_train_loader(train_caption_pkl_path, feature_h5_path, batch_size)
total_step = len(train_loader)

# 准备一下验证用的ground-truth
reference_json_path = '{0}.json'.format(test_reference_txt_path)
reference = COCO(reference_json_path)

# 开始训练模型
best_meteor = 0
loss_count = 0
for epoch in range(num_epochs):
    epsilon = max(0.6, ss_factor / (ss_factor + np.exp(epoch / ss_factor)))
    print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
    log_value('epsilon', epsilon, epoch)
    for i, (videos, captions, cap_lens, video_ids) in enumerate(train_loader, start=1):
        # 构造mini batch的Variable
        videos = Variable(videos)
        targets = Variable(captions)
        if use_cuda:
            videos = videos.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs, video_encoded = banet(videos, targets, epsilon)
        # 因为在一个epoch快要结束的时候，有可能采不到一个刚好的batch
        # 所以要重新计算一下batch size
        bsz = len(captions)
        # 把output压缩（剔除pad的部分）之后拉直
        outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
        outputs = outputs.view(-1, vocab_size)
        # 把target压缩（剔除pad的部分）之后拉直
        targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        log_value('loss', loss.data[0], epoch * total_step + i)
        loss_count += loss.data[0]
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or bsz < batch_size:
            loss_count /= 10 if bsz == batch_size else i % 10
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                  (epoch, num_epochs, i, total_step, loss_count,
                   np.exp(loss_count)))
            loss_count = 0
            tokens = banet.decoder.sample(video_encoded)
            tokens = tokens.data[0].squeeze()
            we = banet.decoder.decode_tokens(tokens)
            gt = banet.decoder.decode_tokens(captions[0].squeeze())
            print('[vid:%d]' % video_ids[0])
            print('WE: %s\nGT: %s' % (we, gt))

    torch.save(banet.state_dict(), banet_pth_path)
    torch.save(optimizer.state_dict(), optimizer_pth_path)
    # 计算一下在val集上的性能并记录下来
    blockPrint()
    banet.eval()
    metrics = evaluate(vocab, banet, test_range, test_prediction_txt_path, reference)
    enablePrint()
    for k, v in metrics.items():
        log_value(k, v, epoch)
        print('%s: %.6f' % (k, v))
        if k == 'METEOR' and v > best_meteor:
            # 备份在val集上METEOR值最好的模型
            shutil.copy2(banet_pth_path, best_banet_pth_path)
            shutil.copy2(optimizer_pth_path, best_optimizer_pth_path)
            best_meteor = v
    banet.train()
