#!/usr/bin/env python
import numpy as np
from ase.io import read
from ase.data.colors import cpk_colors
import sys
from collections import defaultdict
import time
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

if len(sys.argv)!=3:
    print 'Call like',sys.argv[0],'<trajectory file> <feature file>'
    sys.exit(0)

system=sys.argv[1].split('/')[-1][0:-5];

feature_matrix = np.load(sys.argv[2])


n_images,n_atoms_pm,n_feature =feature_matrix.shape

data = read(sys.argv[1]+'@:')
n_samples = len(data)
assert n_samples == n_images, "data and features don't contain the same number of structures"

target = np.array([a.get_potential_energy() for a in data])

type_vec=data[0].get_atomic_numbers()
types=sorted(list(set(type_vec)))
type_mask=[[i for i,a in enumerate(type_vec) if a==t] for t in types]

var=np.var(target)*len(target)


nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.1]
rect_histy = [left_h, bottom, 0.1, height]

# start with a rectangular Figure
plt.figure(1, figsize=(12, 12))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
binwidth = 0.25

    
for i in range(len(types)):
    print i,
    x=feature_matrix[:,type_mask[i],0].flatten()
    y=feature_matrix[:,type_mask[i],1].flatten()
    color=np.append(cpk_colors[types[i]],0.25)
    print color
    axScatter.scatter(x, y,color=color)
    xhist,xedges=np.histogram(x,bins=100,normed=True)
    yhist,yedges=np.histogram(y,bins=100,normed=True)
    axHistx.bar((xedges[:-1]+xedges[1:])/2, xhist*np.diff(xedges),(xedges[1:]-xedges[:-1]),color=cpk_colors[types[i]],alpha=0.5,edgecolor=cpk_colors[types[i]])
    axHisty.barh((yedges[:-1]+yedges[1:])/2,yhist*np.diff(yedges),np.diff(yedges),color=cpk_colors[types[i]],alpha=0.5,edgecolor=cpk_colors[types[i]])
   
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())
axHistx.yaxis.set_major_locator(plt.MaxNLocator(4))
axHisty.xaxis.set_major_locator(plt.MaxNLocator(4))
axHisty.xaxis.tick_top()
axHisty.xaxis.set_ticks_position('both')
plt.setp( axHisty.xaxis.get_majorticklabels(), rotation=70 )

plt.savefig('fscatter_{}_{}.png'.format(system,int(time.time())))

plt.show()
        

