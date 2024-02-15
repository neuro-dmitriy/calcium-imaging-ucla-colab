import matplotlib.pyplot as plt
import numpy as np

import cv2
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo
from caiman.utils.visualization import nb_inspect_correlation_pnr
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import inspect_correlation_pnr

import os
os.chdir('C:/Users/dm/anaconda3/Demo Notebooks/')

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI
    
# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

#%%
    fnames = ['0_piecewise_500.avi']
    ds=1
    
    print('*** laoding vid and motion correct (if on) ***')
    # motion correction parameters
    motion_correct = False            # flag for performing motion correction
    
    pw_rigid   = False                # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt  = (int(12*ds), int(12*ds))          # size of high pass spatial filtering, used in 1p data
    max_shifts = (int(12*ds), int(12*ds))          # maximum allowed rigid shift
    border_nan = 'copy'               # replicate values along the boundaries
    
    mc_dict = {
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'border_nan': border_nan
    }
    
    opts = cnmf.params.CNMFParams(params_dict=mc_dict)
    
    if motion_correct:
    
        # run motion correction
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
    
        fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                                   border_to_0=0, dview=dview)
        
    else:
        fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                                   border_to_0=0, dview=dview)
    
    
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    

#%% correlation & PNR images
    gSig = (int(8*ds), int(8*ds))                                    # expected half size of neurons
    gSiz = (int(16*ds),int(16*ds))                                  # half size for neuron bounding box
    
    cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    
    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    
    # set THESE according to the output from the function above ^
    min_pnr = 10
    min_corr = 0.8

#%% CNMFE pars
    print('*** setting CNMF pars ***')
    rf = int(48*ds)                                # half size of each patch
    stride = int(24*ds)                            # amount of overlap between patches     
    tsub = 2                                       # time downsampling factor   
    ssub = int(2*ds)                               # spatial downsampling factor   
    decay_time = 0.4                               # length of typical transient (in seconds)
    fr = 20                                        # imaging rate (Hz) 
    p = 1                                          # order of AR indicator dynamics
    min_SNR = 1.5                                  # minimum SNR for accepting new components
    rval_thr = 0.85                                # correlation threshold for new component inclusion
    merge_thr = 0.8                                # merging threshold
    K = None                                       # initial number of components
    
    # cnmfe_dict = {'fnames': fnames,
    #               'fr': fr,
    #               'decay_time': decay_time,
    #               'method_init': 'corr_pnr',
    #               'gSig': gSig,
    #               'gSiz': gSiz,
    #               'rf': rf,
    #               'stride': stride,
    #               'p': p,
    #               'low_rank_background': False,
    #               'del_duplicates': True,
    #               'nb': 1,
    #               'nb_patch': 0,
    #               'tsub': tsub,
    #               'ssub': ssub,
    #               'min_SNR': min_SNR,
    #               'min_pnr': min_pnr,
    #               'min_corr': min_corr,
    #               'bas_nonneg': False,
    #               'center_psf': True,
    #               'rval_thr': rval_thr,
    #               # 'only_init': True,
    #               'merge_thr': merge_thr,
    #               'K': K}
    
    
    cnmfe_dict = {'fnames': fnames,
                  'fr': fr,
                  'decay_time': decay_time,
                  'method_init': 'corr_pnr',
                  'gSig': gSig,
                  'gSiz': gSiz,
                  'rf': rf,
                  'stride': stride,
                  'p': p,
                  'nb': 0,
                  'ssub': ssub,
                  'min_SNR': min_SNR,
                  'min_pnr': min_pnr,
                  'min_corr': min_corr,
                  'bas_nonneg': False,
                  'center_psf': True,
                  'rval_thr': rval_thr,
                  'only_init': True,
                  'merge_thr': merge_thr,
                  'K': K}
    
    opts.change_params(cnmfe_dict);


#%%
    print('*** running CNMF ***')
    from time import time
    t1 = -time()
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)
    t1 += time()
    print('*** done ***')


#%% DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5           # adaptive way to set threshold on the transient size
    r_values_min = 0.85     # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': True})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

#%%
    cnm.estimates.plot_contours(img=pnr, idx=cnm.estimates.idx_components)
    cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)
    cnm.estimates.view_components(images, idx=cnm.estimates.idx_components_bad)
    
    # mov = cnm.estimates.play_movie(images, magnification=1, include_bck=False, gain_res=2)

# %% STOP SERVER
    cm.stop_server(dview=dview)


# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
