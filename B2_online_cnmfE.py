import matplotlib.pyplot as plt
import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo
from caiman.utils.visualization import nb_inspect_correlation_pnr

import os
os.chdir('C:/Users/dm/anaconda3/Demo Notebooks/')


#%%
def main():
    pass # For compatibility between running under Spyder and the CLI
    
    #%% server
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)
    
    #%% set fnames (.avi file list)
    max_vid_number = 0
    
    fnames = []
    for i in range(max_vid_number+1):
        filename = str(i) + ".avi"
        fnames.append(filename)
    
    
    #%% online parameters
    
    gSig_filt = (12,12)                            # 12 gave best motion correction for 0.avi
    max_shifts_online = 20
    
    gSig = (8, 8)                                  # expected half size of neurons
    gSiz = (16, 16)                                # half size for neuron bounding box   
    
    min_pnr = 6
    min_corr = 0.6
    min_SNR = 1                                    # minimum SNR for accepting new components
    rval_thr = 0.8                                 # correlation threshold for new component inclusion
    merge_thr = 0.8                                # merging threshold
    K = None                                       # initial number of components
    
    rf = 48                                        # half size of each patch
    stride = 48                                    # amount of overlap between patches       
    overlaps = 24
    
    fr = 20                                        # imaging rate (Hz) 
    decay_time = 0.4                               # length of typical transient (in seconds)
    p = 1                                          # order of AR indicator dynamics
    
    ds_factor = 1                                  # spatial downsampling factor (during online processing)         
    ssub = 1                                       # spatial downsampling factor (during initialization)
    ssub_B = 2                                     # background downsampling factor (use that for faster processing)
    sniper_mode = True                             # flag using a CNN to detect new neurons (o/w space correlation is used)
    init_batch = 300                               # number of frames for initialization (presumably from the first file)
    expected_comps = 500                           # maximum number of expected components used for memory pre-allocation (exaggerate here)
    dist_shape_update = False                      # flag for updating shapes in a distributed way
    min_num_trial = 5                              # number of candidate components per frame     
    K = 60                                         # initial number of components
    epochs = 2                                     # number of passes over the data
    show_movie = True                             # show the movie with the results as the data gets processed
    use_corr_img = True                            # flag for using the corr*pnr image when searching for new neurons (otherwise residual)
    
    online_dict = {
                    'fnames': fnames,
                    'fr': fr,
                    'decay_time': decay_time,
                    'rf': rf,
                    'stride': stride,
                    'overlaps': overlaps,
                    'p': p,
                    'nb': 0,
                    'min_SNR': min_SNR,
                    'min_pnr': min_pnr,
                    'min_corr': min_corr,
                    'rval_thr': rval_thr,
                    'only_init': True,
                    'merge_thr': merge_thr,
                    'K': K,
                    'nb': 0,
                    'ssub': ssub,
                    'ssub_B': ssub_B,
                    'ds_factor': ds_factor,                                   # ds_factor >= ssub should hold
                    'gSig_filt': gSig_filt,
                    'gSig': gSig,
                    'gSiz': gSiz,
                    'bas_nonneg': False,
                    'center_psf': True,
                    'max_shifts_online': max_shifts_online,
                    'motion_correct': True,
                    'init_batch': init_batch,
                    'only_init': True,
                    'method_init': 'corr_pnr',
                    'init_method': 'cnmf',
                    'normalize_init': False,
                    'update_freq': 200,
                    'expected_comps': expected_comps,
                    'sniper_mode': sniper_mode,                               # set to False for 1p data       
                    'dist_shape_update' : dist_shape_update,
                    'min_num_trial': min_num_trial,
                    'epochs': epochs,
                    'use_corr_img': use_corr_img,
                    'show_movie': show_movie
                    }
    
    online_opts = cnmf.params.CNMFParams(params_dict=online_dict);
    
    #%% RUN ONLINE CNMFE
    cnm_online = cnmf.online_cnmf.OnACID(params=online_opts, dview=dview)
    cnm_online.fit_online()
    
    #%% save results to a .mat file
    online_dict['K']=np.nan
    data = {
        'online_dict':          online_dict,
        'A':                    cnm_online.estimates.A,                    # matrix of spatial components (d x K)
        'C':                    cnm_online.estimates.C,                    # matrix of temporal components (K x T)
        'S':                    cnm_online.estimates.S,                    # matrix of merged deconvolved activity (spikes) (K x T)
        'YrA':                  cnm_online.estimates.YrA,                  # matrix of spatial component filtered raw data, after all contributions have been removed
        }
    
    import scipy.io as sio
    sio.savemat('OnACID_CNMFE_ results.mat', {'data': data})
    
    #%% plot components
    cnm_online.estimates.nb_view_components(denoised_color='red');
    cnm_online.estimates.plot_contours_nb()
    
    
# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
