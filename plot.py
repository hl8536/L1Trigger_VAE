import numpy as np
import h5py
import math
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Layer, ReLU, LeakyReLU, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import argparse
import importlib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

plt.rcParams.update({'font.size': 20})

# Loading in the data
bkg_loss = np.load("plotting_data/vae_bkg_loss.npy")
hinv_loss = np.load("plotting_data/vae_hinv_loss.npy")
hllpb_loss = np.load("plotting_data/vae_hllpb_loss.npy")
vqq_loss = np.load("plotting_data/vae_vqq_loss.npy")

bkg_latent = np.load("plotting_data/vae_bkg_latent.npy")
hinv_latent = np.load("plotting_data/vae_hinv_latent.npy")
hllpb_latent = np.load("plotting_data/vae_hllpb_latent.npy")
vqq_latent = np.load("plotting_data/vae_vqq_latent.npy")

labels = ["Bkg", "HtoInv", "VToQQ","Hto2LLPto4b"]
losses = np.asarray([bkg_loss, hinv_loss, vqq_loss,hllpb_loss])

bin_size=25

plt.figure(figsize=(10,8))
#plt.hist(losses[0], bins=bin_size, label=labels[0], density = True, histtype='step', fill=False, linewidth=1.5)
for i, label in enumerate(labels):
    plt.hist(losses[i], bins=bin_size, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
plt.yscale('log')
#plt.xscale('log')
plt.xlabel("VAE Loss")
plt.ylabel("Probability (a.u.)")
plt.grid(True)
plt.title('VAE loss')
plt.legend(loc='best')
plt.savefig("figures/vae_loss.png")

target_background = np.zeros(losses[0].shape[0])

plt.figure(figsize=(8,6))
for i, label in enumerate(labels):
    if i == 0: continue # background events
    
    trueVal = np.concatenate((np.ones(losses[i].shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((losses[i], losses[0]))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
    
    plt.plot(fpr_loss, tpr_loss, "-", label='AUC = %.2f%%'%(auc_loss*100.), linewidth=1.5)
    
    plt.semilogx()
    #plt.semilogy()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
plt.axvline(0.001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection performance
#plt.title("VAE ROC")
plt.savefig("figures/vae_ROC.png")

# Precision Recall Plot
target_background = np.zeros(losses[0].shape[0])

plt.figure(figsize=(8,6))
for i, label in enumerate(labels):
    if i == 0: continue # background events
    
    trueVal = np.concatenate((np.ones(losses[i].shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((losses[i], losses[0]))

    precision, recall, thresholds = precision_recall_curve(trueVal, predVal_loss)

    auc_loss = auc(recall, precision)
    
    plt.plot(recall, precision, "-", label='%s (auc = %.2f%%)'%(label,auc_loss*100.), linewidth=1.5)
    
    #plt.semilogx()
    #plt.semilogy()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    #plt.legend(loc='center right')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
#plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
plt.axhline(0.5,linestyle='dashed',color='0.75')
#plt.axvline(0.01, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection performance
#plt.title("VAE Precision-Recall")
plt.savefig("figures/vae_pr.png")

# Plotting the Latent variables

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
#ax.set_xscale('log')
#ax.set_yscale('log')

ax.scatter(vqq_latent[:,0],vqq_latent[:,1],c="cyan",label="VToQQ",s=10)
ax.scatter(bkg_latent[:,0],bkg_latent[:,1],c="red",label="BKG",s=10)
ax.scatter(hllpb_latent[:,0],hllpb_latent[:,1],c="orange",label="Hto2LLPto4b",alpha=0.5,s=1)
ax.scatter(hinv_latent[:,0],hinv_latent[:,1],c="blue",label="HtoInv",alpha=0.5,s=1)
ax.legend(loc='best')
#ax.set_ylim([-50,50])
ax.set_title("Latent Space Representation")
plt.savefig("figures/vae_latent_space.png")