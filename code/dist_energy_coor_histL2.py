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
from mmutils import cluster

if len(sys.argv)!=4:
    print 'Plot coorlation between mean cluster distance and energy with histograms'
    print 'Call like',sys.argv[0],'<number of clusters> <trajectory file> <feature file>'
    sys.exit(0)

n_clusters=int(sys.argv[1])

parts=sys.argv[3].split("_");

system=parts[-4]

version='L2'
feature_matrix = np.load(sys.argv[3])

#feature_matrix=feature_matrix[:,12:]

n_images,n_atoms,n_features =feature_matrix.shape

data = read(sys.argv[2]+'@:')
n_samples = len(data)
y = np.array([a.get_potential_energy() for a in data])/n_atoms


assert n_samples == n_images, "data and features don't contain the same number of structures"

type_vec=data[0].get_atomic_numbers()

types=sorted(list(set(type_vec)))
type_mask=[[i for i,a in enumerate(type_vec) if a==t] for t in types]


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['text.latex.preamble']=['\usepackage{bm}']
plt.rcParams['text.latex.preamble']=['\usepackage{xfrac}']
plt.rcParams['ps.usedistiller'] = 'xpdf'

plt.rc('font',**{'family':'sans-serif',
             'sans-serif':['Helvetica']})

plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

params={'lines.linewidth':1.,
        'legend.fontsize':9,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'axes.labelsize':9,
        'axes.linewidth':0.5}

plt.rcParams.update(params)

scale=0.27



mlst = feature_matrix.reshape((n_images*n_atoms,n_features))
centroids= cluster(feature_matrix,n_clusters)

dist=np.array([np.sqrt(np.min(np.sum((f-centroids)**2,axis=1))) for f in mlst])
res = np.sum(dist.reshape([n_images,n_atoms]),axis=1)


nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.2, 0.65
bottom, height = 0.15, 0.65
bottom_h = bottom+height+0.03
left_h = left + width + 0.03

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.1]
rect_histy = [left_h, bottom, 0.1, height]

# start with a rectangular Figure
plt.figure(1, figsize=(3.25, 3.25))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
binwidth = 0.25
x=res
color='blue'
axScatter.scatter(x, y,color=color,s=30*scale*scale)
axScatter.set_xlabel("Total cluster distance")
axScatter.set_ylabel("Energy/Atom")
plt.figtext(left+0.05,bottom+height-2*0.05,r'$N_c={}$'.format(n_clusters),fontsize=9)
xhist,xedges=np.histogram(x,bins=100,normed=True)
yhist,yedges=np.histogram(y,bins=100,normed=True)
axHistx.bar((xedges[:-1]+xedges[1:])/2, xhist*np.diff(xedges),(xedges[1:]-xedges[:-1]),edgecolor = "none")
axHisty.barh((yedges[:-1]+yedges[1:])/2,yhist*np.diff(yedges),np.diff(yedges),edgecolor = "none")
   
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())
axHistx.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
axHisty.xaxis.set_major_locator(plt.LinearLocator(numticks=3))
axHisty.xaxis.tick_top()
axHisty.xaxis.set_ticks_position('both')
plt.setp( axHisty.xaxis.get_majorticklabels(), rotation=70 )

plt.savefig('decoorhist{}_{}_{}.pdf'.format(version,n_clusters,sys.argv[2].split('/')[-1][:-5]))

plt.show()
        

