import matplotlib.pyplot as plt
import numpy as np
import cv2
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect

from scipy.io import savemat
import time
import os


def main():
    pass # For compatibility between running under Spyder and the CLI
    
    # %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)
    
    
    #%% get file names
    print('*** prepping file list ***')
    
    min_vid_number = 0      # load videos starting at this index
    max_vid_number = 20     # to this index (0 and 0 would just load 0.avi)
    
    # this folder contains all the avi files, from 0.avi up to ##.avi
    load_fold = 'C:/Users/dm/anaconda3/Demo Notebooks/JL72 26/MC data/'
    
    # this variable crops the video during MotionCorrect procedure
    crop_vid = (slice(0,500),slice(100,600))
    
    # where to save the motion correction files
    save_fold = load_fold + '/MC data/'
    if not(os.path.isdir(save_fold)):
        os.mkdir(save_fold)
    
    fnames = []
    for i in range(min_vid_number,max_vid_number+1):
        fname = load_fold + str(i) + ".avi"
        fnames.append(fname)
    
    #%% striped frames
    # https://groups.google.com/g/miniscope/c/pr7PMiDbK68/m/M5mfHwBUDgAJ
    # # # # # TLDR: there is no good post-hoc fix for bad frames
    # # # # # !! It should be possible to prevent this from happening !!
    # "When I [Daniel Aharoni] have time I will keep trying to track down the
    #  source of this issue but as Zeeshan said, it very likely comes from an
    #  unstable U.FL connection on the V4 Miniscope. Adding some silicone or
    #  epoxy between the coax cable and PCB can help stability this. Your U.FL
    #  connector might also be loose, in which case you should replace it.
    #  Finally, you could try removing the U.FL connector on the PCB and
    #  directly soldering the coax cable to the PCB. Definitely a commutator
    #  can also cause this issue as the slip ring connections inside the
    #  commutator can also add some intermittent instability."
    
    # one file at a time to avoid RAM problems
    ALL_BAD_FRAMES=[]
    stripe_vids=[]
    for fnum,fname in enumerate(fnames):
        
        print('checking for stripes in '+fname)
        images=cm.load(fname)
        
        # detect striped frames
        bad_frames = np.zeros(images.shape[0], dtype=bool)
        max_diff = np.zeros(images.shape[0], dtype=float)
        for i,frame in enumerate(images):
            max_diff[i] = np.max(np.mean(np.abs(np.diff(frame.astype(float),axis=0)),axis=1))
            # the line above gets the max difference between neighboring rows of each frame
            # horizontal stripes in frame = large difference between rows
        bad_frames = max_diff>4     # 4 was a good threshold for the tested vids
        
        ALL_BAD_FRAMES.append(bad_frames)
        
        # replace striped frames with most recent good frame
        if any(bad_frames):
            
            print('*** frames with stripes found ***')
            indexes = np.where(bad_frames)[0]
            print(', '.join(map(str, indexes)))
            print('replacing bad frames with last good frame')
            
            # Iterate over the bad indexes
            for index in indexes:
                # Find the most recent good frame by searching backwards from the bad frame
                if index>0:
                    for i in range(index - 1, -1, -1):
                        if not bad_frames[i]:
                            # Copy the most recent good frame over the bad frame
                            images[index] = images[i]
                            good_frame = images[i]      # keep most recent good frame in memory
                            break
                else:
                    bad_frames[0]=False
                    images[0]=good_frame
                    # ^ as written, MUST have the first frame of first video file be good
            
            # save "stripe corrected" video
            stripe_save_name = save_fold + os.path.splitext(os.path.basename(fname))[0] + '_noStripes.avi'
            fps = 20
            frame_height = images.shape[1]
            frame_width = images.shape[2]
            frame_size = (frame_height, frame_width)
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            out = cv2.VideoWriter(stripe_save_name, fourcc, fps, frame_size, isColor=False)
            for i in range(images.shape[0]):
                frame = np.uint8(images[i, :, :])
                out.write(frame)
            out.release()
            
            # update fnames for the motion-correction step
            fnames[fnum]=stripe_save_name
            stripe_vids.append(stripe_save_name)
            
        else:
            print('*** no striped-frame artifacts found ***')
    
    
    ALL_BAD_FRAMES = np.concatenate(np.asarray(ALL_BAD_FRAMES),axis=0)
    del images,good_frame,bad_frames      # remove some variables to keep ram free
    
    #%% MC pars
    print('*** setting pars ***')
    mc_dict = {
        'gSig_filt'           : (8,8),      # px of spacial frequency to extract from image - 6-12 seem good for our data
        
        'niter_rig'           : 2,          # perform multiple passes of rigid MC
        'max_shifts'          : (50,50),    # 50 to be safe? doesn't seem to increase computation length too much
        
        'indices'             : crop_vid,   # indices auto-crops the video during the MC procedure
        
        'pw_rigid'            : True,      # !!! pw is turned off - but if rigid is not good enough, turn it on
        'max_deviation_rigid' : 10,
        'strides'             : (100,100),  # patch size for PW MC
        'overlaps'            : (50,50),    # patch overlap size for PW MC
    }
    
    opts = cnmf.params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    
    
    #%% MC correction
    print('*** running MC ***')
    start_time = time.time()
    mc.motion_correct(save_movie=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    
    
    #%% make plots
    print('*** making plots ***')
    
    plt.figure(figsize = (10,5))
    
    plt.subplot(2, 1, 1)
    if mc_dict['pw_rigid']:
        plt.plot(mc.x_shifts_els)
    plt.plot(np.asarray(mc.shifts_rig)[:,0],color='black',lw=1)
    plt.ylabel('x shifts (pixels)')
    index_points = np.where(ALL_BAD_FRAMES)[0]
    plt.plot(index_points, np.zeros_like(index_points), 'r.', markersize=1)   # indicate bad frames
    
    plt.subplot(2, 1, 2)
    if mc_dict['pw_rigid']:
        plt.plot(mc.y_shifts_els)
    plt.plot(np.asarray(mc.shifts_rig)[:,1],color='black',lw=1)
    plt.ylabel('y_shifts (pixels)')
    plt.xlabel('frames')
    plt.show()
    index_points = np.where(ALL_BAD_FRAMES)[0]
    plt.plot(index_points, np.zeros_like(index_points), 'r.', markersize=2)
    
    plt.savefig(save_fold+str(min_vid_number)+'-'+str(max_vid_number)+'_MCrigid_shifts.pdf', format='pdf')
    
    
    #%% save data .mat file
    mcSave = {  'shifts_rig'    :   np.asarray(mc.shifts_rig),
                'bad_frames'    :   ALL_BAD_FRAMES   }
    
    savemat(save_fold+str(min_vid_number)+'-'+str(max_vid_number)+'_MCdata.mat', mcSave)
    
    
    #%% write motion corrected video file
    if mc_dict['pw_rigid']:
        fnames_mem = mc.fname_tot_els
    else:
        fnames_mem = mc.fname_tot_rig
    
    # load the memmap files one at a time to avoid running out of RAM
    # ... in order to make one motion corrected .avi video
    fps = 200       # to watch the file at x10 speed
    frame_height = crop_vid[0].stop-crop_vid[0].start
    frame_width = crop_vid[1].stop-crop_vid[1].start
    frame_size = (frame_height, frame_width)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    
    for fnum,fname in enumerate(fnames_mem):
        vid = cm.load(fname)
        
        orig_fname = fnames[fnum]
        save_name = os.path.basename(orig_fname)
        save_name = save_name.replace('_noStripes', '').replace('.avi','_MC.avi')
        print('*** saving '+save_name+' ***')
        out = cv2.VideoWriter(save_fold+save_name, fourcc, fps, frame_size, isColor=False)
        for i in range(vid.shape[0]):
            frame = np.uint8(vid[i, :, :])
            out.write(frame)
        out.release()
    
    
    #%% STOP SERVER    
    cm.stop_server(dview=dview)
    
    #%% file cleanup
    # delete the "intermediate" memmap files and noStripe videos
    # (otherwise, for every raw #.avi, ~1.4GB gets occupied)
    
    del vid     # release the last loaded memmap file
    for fname in fnames_mem+stripe_vids:
        try:
            os.remove(fname)
        except:
            print('could not delete '+fname)


if __name__ == "__main__":
    main()