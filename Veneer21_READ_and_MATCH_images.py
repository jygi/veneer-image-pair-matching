#%% IMPORT LIBRARIES
import os
import pandas as pd
import cv2
import random
import numpy as np
from itertools import combinations

# WORKING DIRECTORY
# workdir = os.getcwd()

# RUN OBJECT AND FUNCTIONS FROM SEPARATE FILES
runfile('Veneer21_FCN_DefectsAnalysis.py', wdir=workdir)
runfile('Veneer21_OBJ_PicGLCM.py', wdir=workdir)

#%% SETTINGS

# Exclude manually detected wrong pairs?
exclusion = 1

# ...Original Full Size Images (source) / p = peeling (wet); d = drying (dry)
pfolder = '.../veneer21_wet/'
dfolder = '.../veneer21_dry/'

# ...Small Size Images Saving Folder (destination)
pPicDest = '.../veneer21_wetSmall/'
dPicDest = '.../veneer21_drySmall/'

# ...GLCM Feature arrays (destination)
pGLCMDest = '.../veneer21_wetGLCM/'
dGLCMDest = '.../veneer21_dryGLCM/'

# ... Folder of matched files
matchingResultsFolder = '.../Matched/'
# OPTIONAL: To skip GLCMs creation, set to zero
glcm_reading = 1


#%% READ IMAGES
# List of image files
pfiles = os.listdir(pfolder)
dfiles = os.listdir(dfolder)

# Test reading an example image and display
img = read_rotate_resize(pfolder+pfiles[0], 
                         side_cuts=5, 
                         top_cuts=5, 
                         crop_thresh=200)
show_im(img)

#%% SAVE SMALL GRAYSCALE VERSIONS OF FULL IMAGES AND EXTRACT GLCM-FEATURES
if glcm_reading == 1:
    # Peeling (wet) images
    for n in range(0,len(pfiles)):
        print(str(n)+'/'+str(len(pfiles)))
        # (Peeling) Read small grayscale image
        im = read_rotate_resize(
                                pfolder+pfiles[n], 
                                side_cuts = 5, 
                                top_cuts = 5, 
                                crop_thresh = 200)
        # (Peeling) Write small grayscale image to file
        cv2.imwrite(
            pPicDest+pfiles[n].split('.')[0]+'.png', 
            im
            )
        # (Peeling) Create a pickled GLCM-array of small grayscale image 
        pickleGLCM_fromSheetImg(
            im,
            pfolder+pfiles[n], 
            savepath=pGLCMDest)
        
    # Drying (dry) images
    for n in range(0,len(dfiles)):
        print(str(n)+'/'+str(len(dfiles)))
        # (Drying) Save small grayscale image
        im = read_rotate_resize(
                                dfolder+dfiles[n], 
                                side_cuts = 5, 
                                top_cuts = 5, 
                                crop_thresh = 200)
        # (Drying) Write small grayscale image to file
        cv2.imwrite(
            dPicDest+dfiles[n].split('.')[0]+'.png', 
            im
            )
        # (Peeling) Create a pickled GLCM-array of small grayscale image 
        pickleGLCM_fromSheetImg(
            im,
            dfolder+dfiles[n], 
            savepath=dGLCMDest)
        
    
#%% CREATE SHEET-OBJECTS
# Read GLCM-files
pglcm_files = os.listdir(pGLCMDest)
dglcm_files = os.listdir(dGLCMDest)

## READ GLCM-ARRAYS TO OBJECTS IN A FOR-LOOP
# Create empty lists for peeling and drying objects
p_objs = []
d_objs = []

for n in range(len(pfiles)):
    print(str(n)+'/'+str(len(pfiles)))

    # (Peeling) Load image and the glcm-data corresponding pickle-file
    img = cv2.imread(pPicDest+pfiles[n], 0)
    pkl_name = pfiles[n].replace('png','pkl')
    glcmData = pd.read_pickle(pGLCMDest+pkl_name)
    # (Peeling) Create picSheet-object with no glcm-data 
    p = create_shPic(img, pfiles[n], calc_glcm=0)
    # (Peeling) Add glcm-data to object
    p.glcmCorr10 = glcmData['c10']
    p.glcmCorr20 = glcmData['c20']
    p.glcmCorr30 = glcmData['c30']
    p.glcmCorr40 = glcmData['c40']
    p.glcmCorr50 = glcmData['c50']
    
    p.glcmDiss10 = glcmData['d10']
    p.glcmDiss20 = glcmData['d20']
    p.glcmDiss30 = glcmData['d30']
    p.glcmDiss40 = glcmData['d40']
    p.glcmDiss50 = glcmData['d50']
    # (Peeling) Append object to list of objects
    p_objs.append(p)
    

for n in range(len(dfiles)):
    print(str(n)+'/'+str(len(dfiles)))

    # (Drying) Load image and the glcm-data corresponding pickle-file
    img = cv2.imread(dPicDest+dfiles[n], 0)
    pkl_name = dfiles[n].replace('png','pkl')
    glcmData = pd.read_pickle(dGLCMDest+pkl_name)
    # (Drying) Create picSheet-object with no glcm-data 
    d = create_shPic(img, dfiles[n], calc_glcm=0)
    # (Drying) Add glcm-data to object
    d.glcmCorr10 = glcmData['c10']
    d.glcmCorr20 = glcmData['c20']
    d.glcmCorr30 = glcmData['c30']
    d.glcmCorr40 = glcmData['c40']
    d.glcmCorr50 = glcmData['c50']
    
    d.glcmDiss10 = glcmData['d10']
    d.glcmDiss20 = glcmData['d20']
    d.glcmDiss30 = glcmData['d30']
    d.glcmDiss40 = glcmData['d40']
    d.glcmDiss50 = glcmData['d50']
    # (Peeling) Append object to list of objects
    d_objs.append(d)

# CREATE LIST COPIES
p_objsTMP = p_objs.copy()
d_objsTMP = d_objs.copy()    

#%% DATA SET CLEANING

if exclusion == 1:
    # Set wrong pair IDs (detected manually based on visual inspection)
    wrongPairs = [2321, 2278, 2486, 910, 2345, 
                  163, 1954, 2568, 2092, 2022, 
                  2131, 2276, 1889, 1890, 374]
    defectiveImages = [2149, 1958, 2018, 712]
    toExclude = set(wrongPairs + defectiveImages)
    
    # Create list for included items for analysis
    toInclude = list(set(range(len(p_objsTMP)))-toExclude)
    
    # Create lists of items
    plist = [p_objsTMP[n].file for n in range(len(p_objsTMP))]
    dlist = [d_objsTMP[n].file for n in range(len(d_objsTMP))]
        
    # Create lists of filenames to keep
    pKeep = ['wet_'+str(toInclude[n])+'.png' for n in range(len(toInclude))]
    dKeep = ['dry_'+str(toInclude[n])+'.png' for n in range(len(toInclude))]
    
    # Write object lists with incuded items only
    p_objs = []
    d_objs = []
    for k in range(len(plist)):
        if plist[k] in pKeep:
            p_objs.append(p_objsTMP[k])
        if dlist[k] in dKeep:
            d_objs.append(d_objsTMP[k])

#%% FIND PAIRS
## 1) Algorithm performance
# Give sample size and number of repetitions (in tuple: (sample size, repetitions))
gridsToUse = [10, 20, 30, 40, 50]
test = [
        (2, 1000), 
        # (4, 1000), 
        # (5, 1000), 
        # (10, 1000), 
        # (20, 1000), 
        # (50, 1000),
        # (100, 100),
        # (200, 50),
        # (300, 50), 
        # (500, 5), 
        # (800, 1), 
        # (1200, 1), 
        # (1800, 1),
        # (len(p_objs), 1),
        ]

# RUN SELECTED PAIR FINDING TEST IN A FOR LOOP
for k in range(len(test)):
    n_items = test[k][0]
    rounds = test[k][1]
    # Set random seed number (for repeatibility / default = 0)
    seed_nr = 0

    ## RUN MATCHER FOR test[k]
    # Reset random number
    random.seed(seed_nr)
    print('Random seed set to '+str(seed_nr)+'\n')
    # Create empty list to store results
    matchRatios = []
    matchData = []
    for n in range(0,rounds):
        print('Matching round '+str(n+1) +' out of '+str(rounds))
        # Sample image data and match
        items = random.sample(range(0, len(p_objs)), n_items)
        ratio, match_df = matcherVeneer21(list(map(d_objs.__getitem__, items)), 
                                          list(map(p_objs.__getitem__, items)),
                                          votesNeeded = 'half',
                                          gridsUsedP1 = gridsToUse,
                                          gridsUsedP2 = gridsToUse)
        # Append matching results to lists
        matchRatios.append(ratio)
        matchData.append(match_df)
    
    # Calculate average matching ratio and standard deviation
    print('\nRESULTS with '+str(n_items)+' to '+str(n_items)+
          ' matching (n='+str(rounds)+')'
          '\n-------------------------------------'
          '\nAvg. matching ratio:\t'+str(np.mean(matchRatios))
          +'\nStd. matching ratio:\t'+str(np.std(matchRatios)))
    
    # Create backup
    matchDataBackUp = matchData.copy()
    
    pd.to_pickle(matchData, 
                 matchingResultsFolder+'MatchData'+\
                 str(test[k][0])+'_n'+str(test[k][1])+'.pkl')

#%%
## 2) Sensitivity analysis for grid optimization
nrGrids = [1,2,3,4,5]
for q in range(0, len(nrGrids)):
    # Provide sample size and number of repetitions (in tuple (size, repetitions))
    test = [(100, 100)]
    # Provide grid space and number of tested grids at once
    availableGrids = [10, 20, 30, 40, 50]
    nGrids = nrGrids[q]
    
    # Create grid combinations to test
    combs = list(combinations(availableGrids, nGrids))
    print('Testing '+str(len(combs))+' combinations')
    
    # Input sample size and number of repetitions
    n_items = test[0][0]
    rounds = test[0][1]
    
    # Run all possible combinations
    for k in range(len(combs)):
        gridsToUse = list(combs[k])
        # Set random seed for repeatibility (default = 0)
        seed_nr = 0
    
        ## Run matcher
        # Reset random number (for repeatibility)
        random.seed(seed_nr)
        print('Random seed set to '+str(seed_nr)+'\n')
        # Create empty list to store results
        matchRatios = []
        matchData = []
        for n in range(0, rounds):
            print('Matching round '+str(n+1) +' out of '+str(rounds))
            # Sample image data and match
            items = random.sample(range(0, len(p_objs)), n_items)
            ratio, match_df = matcherVeneer21(list(map(d_objs.__getitem__, items)), 
                                              list(map(p_objs.__getitem__, items)),
                                              votesNeeded = 'half',
                                              gridsUsedP1 = gridsToUse,
                                              gridsUsedP2 = gridsToUse
                                              )
            # Append matching results to lists
            matchRatios.append(ratio)
            matchData.append(match_df)
        
        # Calculate average matching ratio and standard deviation
        print('\nRESULTS with '+str(n_items)+' to '+str(n_items)+
              ' matching (n='+str(rounds)+')'
              '\n-------------------------------------'
              '\nAvg. matching ratio:\t'+str(np.mean(matchRatios))
              +'\nStd. matching ratio:\t'+str(np.std(matchRatios)))
        
        # Create backup
        matchDataBackUp = matchData.copy()
        # Create name for saving
        savename = 'grids_'+str(nGrids)+'_sensitivity_'+\
            str(combs[k]).replace(', ','_').replace(')','').replace('(','')+\
                '.pkl'
        # Create pickle of the sensitivity analysis
        pd.to_pickle(matchData, savename)

