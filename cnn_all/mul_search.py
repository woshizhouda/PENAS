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

    self.alphas_normal = {"opt0":np.zeros((k, num_ops)),
         "epoch0":np.zeros((k, num_ops)),
         "accurcy0":np.zeros((k, num_ops))
         }
    for i in range(1,20):
        self.alphas_normal.update({'opt'+str(i):np.zeros((k,num_ops))})
        self.alphas_normal.update({'epoch'+str(i):np.zeros((k,num_ops))})
        self.alphas_normal.update({'accurcy'+str(i):np.zeros((k,num_ops))})
    
    for i in range(20):    
        for x in range(k):
            for j in range(num_ops):
                self.alphas_normal["opt"+str(i)][x][j]=1/7.0
            self.alphas_normal["opt"+str(i)][x][0]=0.0
        
    self.alphas_reduce = {"opt0":np.zeros((k, num_ops)),
         "epoch0":np.zeros((k, num_ops)),
         "accurcy0":np.zeros((k, num_ops))
         }
    
    for i in range(1,20):
        self.alphas_reduce.update({'opt'+str(i):np.zeros((k,num_ops))})
        self.alphas_reduce.update({'epoch'+str(i):np.zeros((k,num_ops))})
        self.alphas_reduce.update({'accurcy'+str(i):np.zeros((k,num_ops))})
    
    for i in range(20):    
        for y in range(k):
            for z in range(num_ops):
                self.alphas_reduce["opt"+str(i)][y][z]=1/7.0
            self.alphas_reduce["opt"+str(i)][y][0]=0.0
        
    self.edge_normal = {"edge0":np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2]),
                        "epoch0":np.zeros((k,1)),
                        "accurcy0":np.zeros((k,1))
        }
    for i in range(1,20):
        self.edge_normal.update({'edge'+str(i):np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2])})
        self.edge_normal.update({'epoch'+str(i):np.zeros((k,1))})
        self.edge_normal.update({'accurcy'+str(i):np.zeros((k,1))})
        
    self.edge_reduce = {"edge0":np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2]),
                        "epoch0":np.zeros((k,1)),
                        "accurcy0":np.zeros((k,1))
        }
    for i in range(1,20):
        self.edge_reduce.update({'edge'+str(i):np.array([0.5,0.5,1/3.0,1/3.0,1/3.0,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2])})
        self.edge_reduce.update({'epoch'+str(i):np.zeros((k,1))})
        self.edge_reduce.update({'accurcy'+str(i):np.zeros((k,1))})


  def update_probability(self,accurcy,genotype):
     
    def updating(a1, a2, b1, b2, num):
         it1 = self.alphas_normal["accurcy"+str(num)][a1][a2]
         it2 = self.alphas_normal["epoch"+str(num)][a1][a2]
         it3 = self.alphas_reduce["accurcy"+str(num)][b1][b2]
         it4 = self.alphas_reduce["epoch"+str(num)][b1][b2]
         count1 = 0
         count2 = 0
         for k in range(8):
             if((it1 > self.alphas_normal["accurcy"+str(num)][a1][k]) & (it2 <= self.alphas_normal["epoch"+str(num)][a1][k]) ):
                count1 += 1
             elif((it1 < self.alphas_normal["accurcy"+str(num)][a1][k]) & (it2 >= self.alphas_normal["epoch"+str(num)][a1][k])):
                count1 -= 1
             if((it3 > self.alphas_reduce["accurcy"+str(num)][b1][k]) & (it4 <= self.alphas_reduce["epoch"+str(num)][b1][k])):
                count2 += 1
             elif((it3 < self.alphas_reduce["accurcy"+str(num)][b1][k]) & (it4 >= self.alphas_reduce["epoch"+str(num)][b1][k])):
                count2 -= 1
                         
         self.alphas_normal["opt"+str(num)][a1][a2] = self.alphas_normal["opt"+str(num)][a1][a2] + self.base*count1
         self.alphas_reduce["opt"+str(num)][b1][b2] = self.alphas_reduce["opt"+str(num)][b1][b2] + self.base*count2
    
    def update(accurcy,weightN,weightR,num):
         start = 0

         for i in range(self._steps):
             for j in range(0, 2):
                    t1,t2 = weightN[j+i*2]
                    t1 = PRIMITIVES.index(t1)
                    t2 = t2 + start
                    self.alphas_normal["accurcy"+str(num)][t2][t1] = (self.alphas_normal["epoch"+str(num)][t2][t1]*self.alphas_normal["accurcy"+str(num)][t2][t1]+accurcy)/(self.alphas_normal["epoch"+str(num)][t2][t1] + 1.0)
                    self.alphas_normal["epoch"+str(num)][t2][t1] += 1
           
                    d1,d2 = weightR[j+i*2]
                    d1 = PRIMITIVES.index(d1)
                    d2 = d2 + start
                    self.alphas_reduce["accurcy"+str(num)][d2][d1] = (self.alphas_reduce["epoch"+str(num)][d2][d1]*self.alphas_reduce["accurcy"+str(num)][d2][d1]+accurcy)/(self.alphas_reduce["epoch"+str(num)][d2][d1] + 1.0)
                    self.alphas_reduce["epoch"+str(num)][d2][d1] += 1
           
                    updating(t2, t1, d2, d1, num)
             
             start = start + i + 2
    for i in range(20):
        update(accurcy, genotype[i].normal, genotype[i].reduce, i)
    
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

    #concat = range(2+self._steps-self._multiplier, self._steps+2)
    concat = [2, 3, 4, 5]
    alphas_normal = {}
    edge_normal = {}
    alphas_reduce = {}
    edge_reduce = {}
    genotype = {}
    for i in range(20):
        alphas_normal["opt"] = self.alphas_normal['opt'+str(i)] 
        alphas_normal["epoch"] = self.alphas_normal['epoch'+str(i)]
        alphas_normal["accurcy"] = self.alphas_normal['accurcy'+str(i)]
        edge_normal["edge"] = self.edge_normal['edge'+str(i)]
        edge_normal["epoch"] = self.edge_normal['epoch'+str(i)]
        edge_normal["accurcy"] = self.edge_normal['accurcy'+str(i)]
        
        alphas_reduce["opt"] = self.alphas_reduce['opt'+str(i)]
        alphas_reduce["epoch"] = self.alphas_reduce['epoch'+str(i)]
        alphas_reduce["accurcy"] = self.alphas_reduce['accurcy'+str(i)]
        edge_reduce["edge"] = self.edge_reduce['edge'+str(i)]
        edge_reduce["epoch"] = self.edge_reduce['epoch'+str(i)]
        edge_reduce["accurcy"] = self.edge_reduce['accurcy'+str(i)]
        
        gene_normal = _parse(alphas_normal,edge_normal)
        gene_reduce = _parse(alphas_reduce,edge_reduce)
        
        genotype[i] = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat)
        
    return genotype
