#!/usr/bin/env python

import numpy as np
from random import random, randrange
from math import ceil, cos, sin, pi
from ase.neighborlist import NeighborList
from ase.ga.utilities import atoms_too_close
from ase.ga.utilities import atoms_too_close_two_sets
from ase.ga.offspring_creator import OffspringCreator
from scipy.optimize import minimize
from ase import Atoms


class ClusterDistanceMutation(OffspringCreator):
    """ ClusterDistanceMutation make a feature vector
        for induvidual atoms then cluster the vectors 
        and find cluster centers. 
        Then minimize the total distance from features vectors 
        to clusters centers by moving the atoms.

        Parameters:

        n_clusters: Number of clusters.
        Rcut: Don't consider neighbours beyone this radius
    """
    def __init__(self, slab,n_top,blmin, n_clusters, n_parents=3,r_cut=11.9, verbose=False):
        OffspringCreator.__init__(self, verbose)
        self.slab=slab
        self.n_top = n_top
        self.blmin = blmin
        self.n_clusters = n_clusters
        self.r_cut =r_cut 
        self.descriptor = 'ClusterDistanceMutation'
        self.min_inputs = n_parents

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(parents)
        if indi is None:
            return indi, 'mutation: cluster distance'
            
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [p.info['confid'] for p in parents]

        return self.finalize_individual(indi), 'mutation: cluster distance'

    def mutate(self, parents):
        def clusterdist(feat, centroids):
            fin=feat.reshape(-1, feat.shape[-1])
            dist=np.array([np.sqrt(np.min(np.sum((f-centroids)**2,axis=1))) for f in fin])
            return np.sum(dist)

        def objective_func(pos):
            image.positions[free,:]=np.reshape(pos,[len(free),3])
            fin = buildFeature(image, self.r_cut, types)
            res=clusterdist(fin,centroids)
            grad=agrad(image,free,self.r_cut,fin,centroids,types)
            return res,grad.flatten()


        """ Does the actual mutation. """
        types=sorted(list(set(parents[0].get_atomic_numbers())))
        f=[]
        for image in parents:
            f+=[buildFeature(image, self.r_cut, types)]             
        centroids =cluster(np.array(f),self.n_clusters)             
        energy=[p.get_potential_energy() for p in parents]
        idx=np.argmin(energy)
        image=parents[idx]
        free=range(len(image)-self.n_top,len(image))

        pos=image.positions[free].flatten()
        pos=minimize(objective_func,pos,jac=True, options={'disp':self.verbose,'maxiter':100}).x             
        res=Atoms(numbers=image.numbers[free],positions=pos.reshape(len(free),3),pbc=image.get_pbc(),cell=image.get_cell())
        
        res.wrap()
        return self.slab+res



def real_distances(image,a, indexs,offset):
    dvecs=image.get_distances(a,indexs,vector=True)
    cell=image.get_cell()
    dvecs+=np.dot(offset,cell)
    return np.sqrt(np.sum(dvecs**2,axis=1))

def real_distance(image,indexs,offset):
     pos=image.get_positions()[indexs]
     cell=image.get_cell()
     return np.sqrt(np.sum((pos[0]-pos[1]+np.dot((offset[0]-offset[1]),cell))**2))


def buildFeature(image, cut, types):
    nl=NeighborList([cut/2.0 for i in range(len(image))], skin=0, self_interaction=False,bothways=True)
    nl.update(image)
    typeidx=np.array([types.index(i) for i in image.get_atomic_numbers()])
    res=np.zeros([len(image),len(types)+1])
    for i,a in enumerate(image):
        indices,offset = nl.get_neighbors(i)
        dist=real_distances(image,i, indices,offset)
        res[i,0:len(types)]= np.bincount(typeidx[indices],weights=np.exp(-dist)*(np.cos(np.pi*dist/cut)+1)/2,minlength=len(types))        
        res[i,len(types)]=types[typeidx[i]]
    return(res)


def choice(choices):
    total = sum(choices)
    r = np.random.uniform(0, total)
    upto = 0
    for c, w in enumerate(choices):
        upto += w
        if upto >= r:
            return c
    assert False, "Shouldn't get here"


def kmeanspp(mlst,n_clusters):
    # the kmeans++ cluster initialization algorithme
    #mlst=np.array(mlst)
    cent=np.array([mlst[np.random.randint(len(mlst))]])

    while len(cent)<n_clusters:
        d=np.array([np.min(np.sum((f-cent)**2,axis=1)) for f in mlst]) 
        #print cent.shape,d.shape,np.random.choice(len(mlst),p=d/np.sum(d))
        cent=np.append(cent,[mlst[choice(d)]],axis=0)
    return cent                          

def cluster(micro,n_clusters):
    cz=n_clusters
    n_images,n_atoms_pm,n_features =micro.shape
    mlst=micro.reshape([n_images*n_atoms_pm,n_features])
    old=np.zeros([n_clusters,n_features])
    cc=np.zeros([n_clusters,1],dtype=int)
    new=kmeanspp(mlst,n_clusters)                          
    
    while not np.allclose(old,new):
        old[:]=new
        cc.fill(1)
        for f in mlst: 
            idx=np.argmin(np.sum((f-old)**2,axis=1))
            new[idx]+=f
            cc[idx]+=1
        new/=cc
        
    if np.count_nonzero(np.bincount([np.argmin(np.sum((f-old)**2,axis=1)) for f in mlst],minlength=n_clusters))<n_clusters:
        print "Empty clusters warning"
    return new   

def agrad(image,indexes,cut,fin,centroids,types):
     
     grad=np.zeros([len(indexes),3])
     fin=fin.squeeze()
     centidx=[np.argmin(np.sum((f-centroids)**2,axis=1)) for f in fin.squeeze()]
     fdist=np.sqrt((np.sum((fin-centroids[centidx])**2,axis=1)))
     dvec=np.array([di/(2.0*fd) if fd>0.0 else di for di,fd in zip((fin-centroids[centidx]),fdist)])
     
     nl=NeighborList([cut/2.0 for i in range(len(image))], skin=0, self_interaction=False,bothways=True)
     nl.update(image)
     typeidx=np.array([types.index(i) for i in image.get_atomic_numbers()])
     cell=image.get_cell()
     pvec=image.positions[indexes]
     for i,a in enumerate(indexes):
         indices,offset = nl.get_neighbors(a)
         dvecs=image.get_distances(a,indices,vector=True)
         dvecs+=np.dot(offset,cell)
         dist=np.sqrt(np.sum(dvecs**2,axis=1))
         svec=np.exp(-dist)*(np.pi*np.sin(np.pi*dist/cut)+cut*np.cos(np.pi*dist/cut)+cut)/(dist*2*cut)
         delta=dvec[a,typeidx[indices]]*svec +dvec[indices,typeidx[a]]*svec
         grad[i]-=np.sum(delta,axis=0)*pvec[i]
         grad[i]+=np.sum(np.expand_dims(delta,axis=1)*(dvecs+pvec[i]),axis=0)
         
     return grad



if __name__ == '__main__':
    
    from ase.io import read
    from ase.visualize import view
    from ase.build import stack
    from ase.constraints import FixAtoms
    from ase.ga.startgenerator import StartGenerator
    from ase.ga.utilities import closest_distances_generator
    from ase.calculators.morse import MorsePotential

    #build test atoms
    pos=np.array([[0,1.96794,8.35973568],[0.0,0.0,8.0],[1.967940,1.96794,8.71947137]])
    tmp0=np.array([pos[0]+i*np.array([0.0,3.936,0.0]) for i in range(0,4)])
    tmp1=np.array([pos[1]+i*np.array([0.0,3.936,0.0]) for i in range(0,4)])
    tmp2=np.array([pos[2]+i*np.array([0.0,3.936,0.0]) for i in range(0,4)])

    slab=Atoms('Ti4O8',positions=np.concatenate((tmp0,tmp1,tmp2)),cell=[[  3.93588,  0.0, 0.0] ,[ 0.0,15.74352, 0.0],[0.0, 0.0, 21.4528356]],pbc=[True,True,False],constraint=FixAtoms(indices=range(12)))
    c = slab.get_cell()
    v1 = np.array((c[0][0], 0., 0.))
    v2 = np.array((0., c[1][1], 0.))
    v3 = np.array((0., 0., 6.))
    p0 = np.array((0., 0., 8.))
    box = [p0, [v1, v2, v3]]
    stoichiometry=5*[22]+10*[8]

    print 'slab', buildFeature(slab, 6, [8,22])

    dmin = closest_distances_generator(atom_numbers = [22, 8],
                                       ratio_of_covalent_radii = 0.6)
    
    sg = StartGenerator(slab = slab, # Generator to generate initial structures
                        atom_numbers = stoichiometry, 
                        closest_allowed_distances = dmin,
                        box_to_place_in = box)

    calc=MorsePotential()
    test=sg.get_new_candidate()
    test.set_calculator(calc)
    test.info['confid']=1
    view(test)
    cd=ClusterDistanceMutation(slab,len(stoichiometry),dmin,5,1,verbose=True)

    res,desc = cd.get_new_individual([test])
    res.set_calculator(calc)
    view(res)
    res.info['confid']=2
    res2,desc2 = cd.get_new_individual([res,test])
    view(res2)
