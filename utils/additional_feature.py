import numpy as np
import matplotlib.pyplot as plt
import glob
from ois import optimal_system
from tqdm import tqdm


def get_p2p(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    ptp_img = np.empty([1, patch_size, patch_size])
    
    
    # get first 3 channel as original must be chosen
    ptp_img[0] = np.ptp(trk_imgs[:3], axis=0)
    
    #print(np.shape(ptp_img))
    return ptp_img       

def get_similarity(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    cosine_sim_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    img2 = trk_imgs[1]
    img3 = trk_imgs[2]
 
    
    cosine_sim_imgs[0] = np.dot(template,img2)/(norm(template)*norm(img2))
    cosine_sim_imgs[1] = np.dot(template,img3)/(norm(template)*norm(img3))
    
    return cosine_sim_imgs

def get_diff_imgs(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    diff_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    
    diff_imgs[0] = trk_imgs[1]-trk_imgs[0]
    diff_imgs[1] = trk_imgs[2]-trk_imgs[0]
    
    return diff_imgs 


def get_subtraction_stamp(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    diff_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    
    diff_imgs[0] = trk_imgs[1]-trk_imgs[0]
    diff_imgs[1] = trk_imgs[2]-trk_imgs[0]
    
    return diff_imgs 

def get_bramich_diff(sequence, in_dims=2):
    patch_size = np.shape(sequence)[1]
    
   
    diff_imgs = np.empty([in_dims-1, patch_size, patch_size])
    
    #template
    tmpl = sequence[0]

    
    for idx in range(np.shape(sequence)[0]-1):
        diff_imgs[idx] = optimal_system(sequence[idx+1], tmpl)[0]#trk_imgs[1]-trk_imgs[0]
    
    return diff_imgs 

