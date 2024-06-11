import numpy as np
import matplotlib.pyplot as plt

folder = './images/'
feat_folder = './features/'

alpha = np.load(feat_folder+'_alpha.npy')
beta = np.load(feat_folder+'_beta.npy')
ent = np.load(feat_folder+'_ent.npy')
pos = np.load(feat_folder+'_pos.npy')
kl = np.load(feat_folder+'_kl.npy')
klb = np.load(feat_folder+'_klb.npy')
rel = np.load(feat_folder+'_rel.npy')
seg = np.load(feat_folder+'_seg.npy')

plt.figure(figsize = (20,20))
plt.imshow(alpha[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_alpha.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(beta[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_beta.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(ent[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_ent.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(pos[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_pos.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(np.log10(kl[:,:]), aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_kl.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(klb[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_klb.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(rel[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_rel.png')
plt.close()

plt.figure(figsize = (20,20))
plt.imshow(seg[:,:], aspect = 'auto', interpolation='nearest', cmap = 'gray')
plt.savefig(folder+'_seg.png')
plt.close()

print('\nImages saved in folder.\n')