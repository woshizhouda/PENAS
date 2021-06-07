import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from operations import *

from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np
#from astor.source_repr import count


class MUL(nn.Module):

  def __init__(self, steps=4, multiplier=4):
    super(MUL, self).__init__()
    self._steps = steps
    self._multiplier = multiplier
    self.base = 0.02
    self._initialize_alphas()
    
  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = {"opt":np.zeros((k, num_ops)),
         "epoch":np.zeros((k, num_ops)),
         "accurcy":np.zeros((k, num_ops))
         }
    for x in range(k):
        for j in range(num_ops):
            self.alphas_normal["opt"][x][j]=1/7.0
        self.alphas_normal["opt"][x][0]=0.0
        
    self.alphas_reduce = {"opt":np.zeros((k, num_ops)),
         "epoch":np.zeros((k, num_ops)),
         "accurcy":np.zeros((k, num_ops))
         }
    for y in range(k):
        for z in range(num_ops):
            self.alphas_reduce["opt"][y][z]=1/7.0
        self.alphas_reduce["opt"][y][0]=0.0
        
    self.edge_normal = {"edge":np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2]),
                        "epoch":np.zeros((k,1)),
                        "accurcy":np.zeros((k,1))
        }
    self.edge_reduce = {"edge":np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2]),
                        "epoch":np.zeros((k,1)),
                        "accurcy":np.zeros((k,1))
        }
#     self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
#     self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
#    
#     self._arch_parameters = [
#       self.alphas_normal,
#       self.alphas_reduce,
#     ]

  def genotype(self):

    def _parse(weights,pros):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights["opt"][start:end].copy()
            for j in range(0, i+2):
                k_best = None
                xuanze = 0.0
                for xu1 in range(8):
                    if(W[j][xu1] > 0):
                        xuanze = xuanze + W[j][xu1]
                for xu2 in range(8):
                    if(W[j][xu2] < 0):
                        W[j][xu2] = 0
                    else:
                        W[j][xu2] = W[j][xu2]/xuanze
                while(1):
                    r = np.random.choice(a=8, size=1, p=W[j])
                    r = r.item()
                    if (r != PRIMITIVES.index('none')):
                        k_best = r
                        break
                    else:
                        continue    
                gene.append((PRIMITIVES[k_best], j))
          
            start = end
            n += 1
        return gene

#     gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
#     gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    gene_normal = _parse(self.alphas_normal,self.edge_normal)
    gene_reduce = _parse(self.alphas_reduce,self.edge_reduce)

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    concat = [2, 3, 4, 5]
     
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotype_edge(self):

    def _parse(weights,pros):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights["opt"][start:end].copy()
            #edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            edges = []
            pro = pros["edge"][start:end].copy()
#         p = []
#         for m in range(end-start):
#             p.append(pro[m])
            choice = 0.0
            for u1 in range(end-start):
                if(pro[u1] > 0):
                    choice = choice + pro[u1]
            for u2 in range(end-start):
                if(pro[u2] < 0):
                    pro[u2] = 0
                else:
                    pro[u2] = pro[u2]/choice
            s = np.random.choice(a=end-start, size=2, replace=False, p=pro)
            s1 = s[0].item()
            s2 = s[1].item()
            edges.append(s1)
            edges.append(s2)
#             while(1):
#                 q = np.random.choice(end-start,1,pro.all())
#                 q = q.item()
#                 if(s != q):
#                     edges.append(q)
#                     break
#                 else:
#                     continue
            for j in edges:
                k_best = None
#           for k in range(len(W[j])):
#             if k != PRIMITIVES.index('none'):
#               if k_best is None or W[j][k] > W[j][k_best]:
#                 k_best = k
                xuanze = 0.0
                for xu1 in range(8):
                    if(W[j][xu1] > xuanze):
                        xuanze = W[j][xu1]
                        k_best = xu1
                          
                gene.append((PRIMITIVES[k_best], j))
          
            start = end
            n += 1
        return gene

#     gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
#     gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    gene_normal = _parse(self.alphas_normal,self.edge_normal)
    gene_reduce = _parse(self.alphas_reduce,self.edge_reduce)

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    concat = [2, 3, 4, 5]
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def update_probability(self,accurcy,genotype):
     
    def updating(a1, a2, b1, b2):
         it1 = self.alphas_normal["accurcy"][a1][a2]
         it2 = self.alphas_normal["epoch"][a1][a2]
         it3 = self.alphas_reduce["accurcy"][b1][b2]
         it4 = self.alphas_reduce["epoch"][b1][b2]
         count1 = 0
         count2 = 0
         for k in range(8):
             if((it1 > self.alphas_normal["accurcy"][a1][k]) & (it2 <= self.alphas_normal["epoch"][a1][k]) ):
                count1 += 1
             elif((it1 < self.alphas_normal["accurcy"][a1][k]) & (it2 >= self.alphas_normal["epoch"][a1][k])):
                count1 -= 1
             if((it3 > self.alphas_reduce["accurcy"][b1][k]) & (it4 <= self.alphas_reduce["epoch"][b1][k])):
                count2 += 1
             elif((it3 < self.alphas_reduce["accurcy"][b1][k]) & (it4 >= self.alphas_reduce["epoch"][b1][k])):
                count2 -= 1
                         
         self.alphas_normal["opt"][a1][a2] = self.alphas_normal["opt"][a1][a2] + self.base*count1
         self.alphas_reduce["opt"][b1][b2] = self.alphas_reduce["opt"][b1][b2] + self.base*count2
    
    def update(accurcy,weightN,weightR):
         start = 0

         for i in range(self._steps):
             for j in range(0, 2):
                    t1,t2 = weightN[j+i*2]
                    t1 = PRIMITIVES.index(t1)
                    t2 = t2 + start
                    self.alphas_normal["accurcy"][t2][t1] = (self.alphas_normal["epoch"][t2][t1]*self.alphas_normal["accurcy"][t2][t1]+accurcy)/(self.alphas_normal["epoch"][t2][t1] + 1.0)
                    self.alphas_normal["epoch"][t2][t1] += 1
           
                    d1,d2 = weightR[j+i*2]
                    d1 = PRIMITIVES.index(d1)
                    d2 = d2 + start
                    self.alphas_reduce["accurcy"][d2][d1] = (self.alphas_reduce["epoch"][d2][d1]*self.alphas_reduce["accurcy"][d2][d1]+accurcy)/(self.alphas_reduce["epoch"][d2][d1] + 1.0)
                    self.alphas_reduce["epoch"][d2][d1] += 1
           
                    updating(t2, t1, d2, d1)
             
             start = start + i + 2
  
    update(accurcy, genotype.normal, genotype.reduce)
    
  def update_probability_edge(self,accurcy,genotype):                            
                      
    def renew():
        start = 0
        end = 2
        for i in range(self._steps):
             da_normal = 0
             da_reduce = 0
             for zh in range(start, end):
                 if(self.edge_normal["edge"][zh] > 0):
                     da_normal += 1
                 if(self.edge_reduce["edge"][zh] > 0):
                     da_reduce += 1
             for j in range(start, end):
                 it1 = self.edge_normal["accurcy"][j]
                 it2 = self.edge_normal["epoch"][j]
                 it3 = self.edge_reduce["accurcy"][j]
                 it4 = self.edge_reduce["epoch"][j]
                 count1 = 0
                 count2 = 0
                 if(da_normal > 2):
                    for k in range(start, end):
                        if((it1 > self.edge_normal["accurcy"][k]) & (it2 <= self.edge_normal["epoch"][k])):
                            count1 += 1
                        elif((it1 < self.edge_normal["accurcy"][k]) & (it2 >= self.edge_normal["epoch"][k])):
                            count1 -= 1
                 if(da_reduce > 2):
                    for c in range(start, end):
                        if((it3 > self.edge_reduce["accurcy"][c]) & (it4 <= self.edge_reduce["epoch"][c])):
                            count2 += 1
                        elif((it3 < self.edge_reduce["accurcy"][c]) & (it4 >= self.edge_reduce["epoch"][c])):
                            count2 -= 1
                         
                 self.edge_normal["edge"][j] = self.edge_normal["edge"][j] + self.base*count1
                 self.edge_reduce["edge"][j] = self.edge_reduce["edge"][j] + self.base*count2
                             
             start = end
             end = end + i + 3
    
    def update(accurcy,weightN,weightR):
         start = 0
         for i in range(self._steps):
             t1,t2 = weightN[i*2]
             t1 = PRIMITIVES.index(t1)
             t2 = t2 + start
             self.edge_normal["accurcy"][t2] = (self.edge_normal["epoch"][t2]*self.edge_normal["accurcy"][t2]+accurcy)/(self.edge_normal["epoch"][t2] + 1.0)
             self.edge_normal["epoch"][t2] += 1
           
             c1,c2 = weightN[i*2+1]
             c1 = PRIMITIVES.index(c1)
             c2 = c2 + start
             self.edge_normal["accurcy"][c2] = (self.edge_normal["epoch"][c2]*self.edge_normal["accurcy"][c2]+accurcy)/(self.edge_normal["epoch"][c2] + 1.0)
             self.edge_normal["epoch"][c2] += 1
           
             d1,d2 = weightR[i*2]
             d1 = PRIMITIVES.index(d1)
             d2 = d2 + start
             self.edge_reduce["accurcy"][d2] = (self.edge_reduce["epoch"][d2]*self.edge_reduce["accurcy"][d2]+accurcy)/(self.edge_reduce["epoch"][d2] + 1.0)
             self.edge_reduce["epoch"][d2] += 1
           
             e1,e2 = weightR[i*2+1]
             e1 = PRIMITIVES.index(e1)
             e2 = e2 + start
             self.edge_reduce["accurcy"][e2] = (self.edge_reduce["epoch"][e2]*self.edge_reduce["accurcy"][e2]+accurcy)/(self.edge_reduce["epoch"][e2] + 1.0)
             self.edge_reduce["epoch"][e2] += 1
             
             start = start + i + 2
  
    update(accurcy, genotype.normal, genotype.reduce)
    renew()
    
  def save(self):
    dic = {"alphas_normal": self.alphas_normal, "alphas_reduce": self.alphas_reduce,
            "edge_normal": self.edge_normal, "edge_reduce": self.edge_reduce}
    torch.save(dic, "EXP/mul.pt")
    
  def load(self):
    dic = torch.load("EXP/mul.pt")
    self.alphas_normal = dic["alphas_normal"]
    self.alphas_reduce = dic["alphas_reduce"]
    self.edge_normal = dic["edge_normal"]
    self.edge_reduce = dic["edge_reduce"]
    
   
   
  def genotype_all(self):

    def _parse(weights,pros):
        gene = []
        n = 2
        start = 0
        edges = []
        for i in range(self._steps):
            end = start + n
            W = weights["opt"][start:end].copy()
            #edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            #edges = []
            pro = pros["edge"][start:end].copy()
#         p = []
#         for m in range(end-start):
#             p.append(pro[m])
            choice = 0.0
            for u1 in range(end-start):
                if(pro[u1] > 0):
                    choice = choice + pro[u1]
            for u2 in range(end-start):
                if(pro[u2] < 0):
                    pro[u2] = 0
                else:
                    pro[u2] = pro[u2]/choice
            s = np.random.choice(a=end-start, size=2, replace=False, p=pro)
            s1 = s[0].item()
            s2 = s[1].item()
            edges.append(s1)
            edges.append(s2)

            start = end
            n += 1
            
        temp = 0
        sta = 0
        tt = 2
        for j in edges:
            if(temp%2 == 0):
                if(temp != 0):
                    sta = sta + tt
                    tt = tt + 1
                    
            fin = sta + j
            W = weights["opt"][fin].copy()
            k_best = None
            xuanze = 0.0
            for xu1 in range(8):
                if(W[xu1] > 0):
                    xuanze = xuanze + W[xu1]
            for xu2 in range(8):
                if(W[xu2] < 0):
                    W[xu2] = 0
                else:
                    W[xu2] = W[xu2]/xuanze
            while(1):
                r = np.random.choice(a=8, size=1, p=W)
                r = r.item()
                if (r != PRIMITIVES.index('none')):
                    k_best = r
                    break
                else:
                    continue          
            gene.append((PRIMITIVES[k_best], j))
            temp = temp + 1
        return gene

#     gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
#     gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    gene_normal = _parse(self.alphas_normal,self.edge_normal)
    gene_reduce = _parse(self.alphas_reduce,self.edge_reduce)

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    concat = [2, 3, 4, 5]
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
