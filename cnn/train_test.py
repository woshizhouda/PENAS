import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Subset

#from model_search import Network
from architect import Architect
from train_test_model_node import Network
from train_test_model import Network_edge
import genotypes
from mul_search import MUL
from genotypes import Genotype


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  #np.random.seed(args.seed)
  torch.cuda.set_device(int(args.gpu))
    
  cudnn.benchmark = True
  #torch.manual_seed(args.seed)
  cudnn.enabled=True
  #torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  mul = MUL()
  mul.load()
#   logging.info("%s", mul.alphas_normal)
#   logging.info("%s", mul.alphas_reduce)
#   logging.info("%s", mul.edge_normal)
#   logging.info("%s", mul.edge_reduce)
#   exit(1)
  #genotype = mul.genotype()
  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  #model = model.cuda()
  
#   criterion = nn.CrossEntropyLoss()
#   criterion = criterion.cuda()
#   model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
#   model = model.cuda()
  mm = torch.load(os.path.join('./search-EXP-20190726-162625', 'weights.pt'))
  #mmm = {k.replace('module.',''):v for k,v in mm.items()}
  #model.load_state_dict(mm)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  #train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=False, transform=valid_transform)

  #train_queue = torch.utils.data.DataLoader(
  #  train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

#   mul = MUL()
#   genotype = mul.genotype()
#   model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
#   model = model.cuda()
  #print(mul.genotype())
  #genotype = mul.genotype()
  #print(genotype)
  #exit(1)
#   criterion = nn.CrossEntropyLoss()
#   criterion = criterion.cuda()
#   model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
#   model = model.cuda()
#   model.load_state_dict(mm)
  bestepoch = 0
  bestacc = 0.0
  for epoch in range(100):
# #     genotype = model.genotype()
# #     logging.info('genotype = %s', genotype)
        #model.drop_path_prob = args.drop_path_prob * epoch / args.epochs  
# #     print(F.softmax(model.alphas_normal, dim=-1))
# #     print(F.softmax(model.alphas_reduce, dim=-1))
        #genotype = mul.genotype()
        #genotype = mul.genotype_edge()
        genotype = mul.genotype_all()
#         gene_normal= [('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3)]
#         concat=[2, 3, 4, 5]
#         gene_reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)]
#         genotype = Genotype(
#             normal=gene_normal, normal_concat=concat,
#             reduce=gene_reduce, reduce_concat=concat)
        logging.info("%s", genotype)
        #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, genotype)
        model = Network_edge(args.init_channels, CIFAR_CLASSES, args.layers, genotype)
        model = model.cuda()
        mod = model.state_dict()
        for key in mm:
            if(key in mod):
                mod[key] = mm[key]
        model.load_state_dict(mod)
#         for key in model.state_dict():
#             logging.info("%s",key)
#         exit(1)
#         criterion = nn.CrossEntropyLoss()
#         criterion = criterion.cuda()
#         model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
#         model = model.cuda()
        #model.load_state_dict(mm)
 
        #model.drop_path_prob = args.drop_path_prob
        logging.info('epoch %d', epoch)
        with torch.no_grad():
            valid_acc = infer(valid_queue, model)
        logging.info('valid_acc %f', valid_acc)
        if(valid_acc > bestacc):
            bestacc = valid_acc
            bestepoch = epoch
        print(bestepoch)
        #mul.update_probability(valid_acc, genotype)
        #mul.update_probability_edge(valid_acc, genotype)
        #break
#    utils.save(model, os.path.join(args.save, 'weights.pt'))
  #mul.save()
  #logging.info("%s", mul.alphas_normal)
  #logging.info("%s", mul.alphas_reduce)
  #logging.info("%s", mul.edge_normal)
  #logging.info("%s", mul.edge_reduce)

#   for epoch in range(200):
#         genotype = mul.genotype_edge()
#         logging.info("%s", genotype)
#         #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, genotype)
#         model = Network_edge(args.init_channels, CIFAR_CLASSES, args.layers, genotype)
#         model = model.cuda()
#         mod = model.state_dict()
#         for key in mm:
#             if(key in mod):
#                 mod[key] = mm[key]
#         model.load_state_dict(mod)
# #         for key in model.state_dict():
# #             logging.info("%s",key)
# #         exit(1)
# #         criterion = nn.CrossEntropyLoss()
# #         criterion = criterion.cuda()
# #         model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
# #         model = model.cuda()
#         #model.load_state_dict(mm)
#  
#         #model.drop_path_prob = args.drop_path_prob
#         logging.info('epoch %d', epoch)
#         with torch.no_grad():
#             valid_acc = infer(train_queue, model)
#         logging.info('valid_acc %f', valid_acc)
#         mul.update_probability_edge(valid_acc, genotype)
#   mul.save()
#   logging.info("%s", mul.edge_normal)
#   logging.info("%s", mul.edge_reduce)

def infer(valid_queue, model):
  #objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()

    logits = model(input)
    #loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    #objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %f %f', step, top1.avg, top5.avg)

  return top1.avg


if __name__ == '__main__':
  main()
