import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import random
import time
from scipy.stats import gamma, entropy
from scipy.special import gammaln, psi

seed = 123  
np.random.seed(seed)

# Definire il parser
def get_args_parser():
    parser = argparse.ArgumentParser('SVM feature computation', add_help = False)
    parser.add_argument('--patch_size', default = (7,14), type = int)
    parser.add_argument('--thrKL', default = 0.005)
    parser.add_argument('--eps', default = 1e-3, type = float)
    parser.add_argument('--trimds', default = 27330, type = float)
    parser.add_argument('--datapath', default = '/data/MCoRDS1_2010_DC8/RG2_MCoRDS1_2010_DC8.pt')
    parser.add_argument('--segpath', default = '/data/MCoRDS1_2010_DC8/SG3_MCoRDS1_2010_DC8.pt')
    parser.add_argument('--savepath', default = './features/')
    parser.add_argument('--images_path', default = './images/')

    return parser

def main(args):
    eps = args.eps
    data = torch.load(args.datapath).numpy()[:,:args.trimds]
    seg = torch.load(args.segpath).numpy()[:,:args.trimds]
    data = -(data*30.703053-248.91953) # Unnormalize

    np.save(args.savepath+'_data.npy', data)
    plt.imshow(data)
    plt.savefig(args.images_path+'_data.png')
    plt.close()

    H,W = data.shape
    h,w = args.patch_size
    print('\nShape of the data: {}'.format(data.shape))
    print('Shape of the patch: {}'.format(args.patch_size))
    print('Data mean and std', data.mean(), data.std())

    # Features are (data, alpha_Gamma, beta_Gamma, KL_RSS, Entropy, VerticalPos, Relational)

    # Noise distribition
    noise = data[-50:,:]
    shape, _, scale = gamma.fit(noise.flatten(), floc=0)
    mean_noise = noise.mean()
    alpha_noise = shape
    beta_noise = 1/(scale+eps)
    print('Alpha/Beta noise:', alpha_noise, beta_noise)

    # VerticalPos
    pos = np.tile(np.linspace(0,1,H)[:,np.newaxis], (1,W))

    shapes = np.zeros_like(data)
    scales = np.zeros_like(data)
    ents = np.zeros_like(data)
    kls = np.zeros_like(data)
    klb = np.zeros_like(data)

    # Pad input
    data = np.copy(np.pad(data, pad_width = ((h//2,h//2),(w//2,w//2)), mode='reflect'))
    print('Shape after padding: {}\n'.format(data.shape))
    Hpad,Wpad = data.shape

    time_1 = time.time()
    for j in range(W):
        if j%100==0:
            print('Processing column:',j)
        for i in range(H):
            patch = data[i:i+h,j:j+w]
            patch = patch.flatten()
            shape, _, scale = gamma.fit(patch, floc=0)
            
            ent = np.log10(entropy(patch))
            alpha = shape
            beta = 1/(scale+eps)
            kl = (alpha_noise - alpha) * psi(alpha_noise)+gammaln(alpha) - gammaln(alpha_noise)+alpha * np.log(beta_noise / beta)+alpha_noise * (beta - beta_noise) / beta_noise

            shapes[i,j] = alpha
            scales[i,j] = beta
            ents[i,j] = ent
            kls[i,j] = kl
        

    klb = kls > args.thrKL*mean_noise
    rel = np.cumsum(klb != 1, axis = 0)
    print('For loops computed in',(time.time()-time_1)/60,'minutes\n')

    # Plot info
    print('Alpha mean/std/shape:', shapes.mean(), shapes.std(),shapes.shape)
    print('Beta mean/std/shape:', scales.mean(), scales.std(),scales.shape)
    print('Ent mean/std/shape:', ents.mean(), ents.std(),ents.shape)
    print('KL mean/std/shape:\n', kls.mean(), kls.std(),kls.shape)

    # Save
    np.save(args.savepath+'_pos.npy', pos)
    np.save(args.savepath+'_alpha.npy', shapes)
    np.save(args.savepath+'_beta.npy', scales)
    np.save(args.savepath+'_ent.npy', ents)
    np.save(args.savepath+'_kl.npy', kls)
    np.save(args.savepath+'_klb.npy', klb)
    np.save(args.savepath+'_rel.npy', rel)
    np.save(args.savepath+'_seg.npy', seg)
    print('Done!')
    pass

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)