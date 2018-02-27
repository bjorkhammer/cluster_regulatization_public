#!/usr/bin/env python
import numpy as np
#import tensorflow as tf
from ase.io import read
import time
import random
import sys
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from ase.neighborlist import NeighborList 
from ase.data import covalent_radii
from collections import Counter
from itertools import combinations_with_replacement

def real_distances(image,a, indexs,offset):
    dvecs=image.get_distances(a,indexs,vector=True)
    cell=image.get_cell()
    dvecs+=np.dot(offset,cell)
    return np.sqrt(np.sum(dvecs**2,axis=1))

def real_distance(image,indexs,offset):
    pos=image.get_positions()[indexs]
    cell=image.get_cell()
    #print 'pos',pos[0],pos[1],pos[0]-pos[1],offset[0]-offset[1]
    #print 'cell',cell,np.dot((offset[0]-offset[1]),cell)
    
    return np.sqrt(np.sum((pos[0]-pos[1]+np.dot((offset[0]-offset[1]),cell))**2))


def buildFeature(data, cut, types, radii):
    res=[]
    combine=combinations_with_replacement(range(len(types)), 2)
    for image in data:
        ires=[]
        nl=NeighborList([cut/2.0 for i in range(len(image))], skin=0, self_interaction=False,bothways=True)
        nl.update(image)
        atype=np.array(image.get_atomic_numbers())
     
        for i,a in enumerate(image):
            ares=np.zeros(len(types)+1)
            indices,offset = nl.get_neighbors(i)
            #dist=image.get_distances(i,indices,mic=True)
            dist=real_distances(image,i, indices,offset)
            rad=radii[types.index(atype[i])]
            for k,d in enumerate(dist):
                ares[types.index(atype[indices[k]])]+=np.exp(-d)*(np.cos(np.pi*d/cut)+1)/2.0             
            ares[len(types)]=atype[i]
                
            ires.append(ares)
            
        #print ires    
        res.append(ires)
    return(np.array(res))
    

Rcut=11.9


if len(sys.argv)!=2:
    print 'Call like',sys.argv[0],'<traj file>'
    sys.exit(0)

#system=sys.argv[1].split("_")[1][:-5]            
system=sys.argv[1].split("/")[-1][:-5]            
data = read(sys.argv[1]+'@:')
n_images=len(data)
n_atoms=len(data[0].get_atomic_numbers())
type_vec=data[0].get_atomic_numbers()
types=sorted(list(set(type_vec)))

type_mask=[[i for i,a in enumerate(type_vec) if a==t] for t in types]

type_freq= Counter(data[0].get_atomic_numbers())

type_freq=[type_freq[t] for t in types]

time=int(time.time())

radii=[covalent_radii[t] for t in types]
fin = buildFeature(data, Rcut, types, radii)

np.save('fgen_{}_{}_{}'.format(system,Rcut,time),fin)


