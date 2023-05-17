#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to get image from bond, angle and dihe connectivities in peptide sequence
@author: poonam pandey
"""
import numpy as np
import os
import shutil
import math, os
from collections import OrderedDict
import glob as gb
from sklearn.preprocessing import normalize
import torch
import torchvision.transforms as transforms
from PIL import Image
if __name__ == "__main__": 
    resAtomtypes=list(np.load('../Dataset/example/resAtomtypes.npy'));
    n = len(resAtomtypes)
    outdir='../Dataset/example/'
    seq_data=np.loadtxt('../Dataset/example/seq.fasta',dtype='str')
    print('The Amino acid sequence is: \n'+seq_data[1])
    atoms=list(np.load('../Dataset/example/atoms.npy'))
    n = len(resAtomtypes)
    bond_Mat = [[0 for x in range(n)] for y in range(n)];
    angle_Mat = [[0 for x in range(n)] for y in range(n)];
    dihe_Mat = [[0 for x in range(n)] for y in range(n)];
    bonds=np.load('../Dataset/example/bond_con.npy');
    for bond in bonds:
        a_num1=resAtomtypes.index(atoms[int(bond.split()[0])-1]);
        a_num2=resAtomtypes.index(atoms[int(bond.split()[1])-1]);
        bond_Mat[a_num1][a_num2]=bond_Mat[a_num1][a_num2]+1;
        bond_Mat[a_num2][a_num1]=bond_Mat[a_num2][a_num1]+1;
    r=np.asarray(bond_Mat)
    angles=np.load('../Dataset/example/angle_con.npy')
    for angle in angles:
        a_num1=resAtomtypes.index(atoms[int(angle.split()[0])-1]);
        a_num2=resAtomtypes.index(atoms[int(angle.split()[1])-1]);
        angle_Mat[a_num1][a_num2]=angle_Mat[a_num1][a_num2]+1;
        angle_Mat[a_num2][a_num1]=angle_Mat[a_num2][a_num1]+1;
    b=np.asarray(angle_Mat)
    dihes=np.load('../Dataset/example/dihe_con.npy')
    for dihe in dihes:
        a_num1=resAtomtypes.index(atoms[int(dihe.split()[0])-1]);
        a_num2=resAtomtypes.index(atoms[int(dihe.split()[1])-1]);
        dihe_Mat[a_num1][a_num2]=dihe_Mat[a_num1][a_num2]+1;
        dihe_Mat[a_num2][a_num1]=dihe_Mat[a_num2][a_num1]+1;
    g=np.asarray(dihe_Mat)
    uatoms1=np.asarray(resAtomtypes)
    uatoms=[];
    for at in uatoms1:
        if not at.startswith(('D')):
            uatoms.append(at)
    uatoms=np.asarray(uatoms)
    n1=len(np.unique(uatoms))
    r1 = [[0 for x in range(n1)] for y in range(n1)];
    b1 = [[0 for x in range(n1)] for y in range(n1)];
    g1 = [[0 for x in range(n1)] for y in range(n1)];

    for x in range(0,n1):
        for y in range(0,n1):
            a_num1=resAtomtypes.index(uatoms[x]); 
            a_num2=resAtomtypes.index(uatoms[y]);
            r1[x][y]=r[a_num1][a_num2]
            b1[x][y]=b[a_num1][a_num2]
            g1[x][y]=g[a_num1][a_num2]
    r1=np.asarray(r1)
    b1=np.asarray(b1)
    g1=np.asarray(g1)
    rgbArray = np.zeros((n1, n1, 3), 'uint8')
    rNew = np.interp(r1, (r1.min(), r1.max()), (0, 255))
    gNew = np.interp(g1, (g1.min(), g1.max()), (0, 255))
    bNew = np.interp(b1, (b1.min(), b1.max()), (0, 255))
    rgbArray[..., 0] = rNew
    rgbArray[..., 1] = bNew
    rgbArray[..., 2] = gNew
    img=Image.fromarray(rgbArray, 'RGB')                      
    img.save(outdir+'/img_seq.jpeg')
        
    