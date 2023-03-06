# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:23:01 2021

@author: savojy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.feature import greycomatrix, greycoprops

#%%

class sheetPic:
    """Create sheet object from defect data of knots 'knot_df'"""
        
    def __init__(self):
        """Provide data of knots as pandas dataframe. Give minimum resolution
        that is taken into account in the analysis [netsizeMin]"""
        self.file = []
        self.time = []
        self.defects = []
        self.pic_bw = []
        self.glcmDiss10 = []
        self.glcmDiss20 = []
        self.glcmDiss30 = []
        self.glcmDiss40 = [] 
        self.glcmDiss50 = []
        self.glcmCorr10 = []
        self.glcmCorr20 = []
        self.glcmCorr30 = []
        self.glcmCorr40 = []
        self.glcmCorr50 = []
        self.fpData = []
        self.fpPic =[]
        self.logGLCM100 = []
        # Temporary slot for testing
        self.tmpVar = []
    
    #GLCM data
    def showGLCM(self, diss_min = 0, diss_max = 30, row_drop = None, 
                 col_drop = None, grid = 30):
        """Display glcm dissimilarity and correlation data as a heatmap 
        representing texture changes in the sheet image"""
        ## Copy data and index if needed
        if grid == 10: 
            data_diss = self.glcmDiss10.copy()
            data_corr = self.glcmCorr10.copy()
        if grid == 20: 
            data_diss = self.glcmDiss20.copy()
            data_corr = self.glcmCorr20.copy()
        if grid == 30: 
            data_diss = self.glcmDiss30.copy()
            data_corr = self.glcmCorr30.copy()
        if grid == 40: 
            data_diss = self.glcmDiss40.copy()
            data_corr = self.glcmCorr40.copy()
        if grid == 50: 
            data_diss = self.glcmDiss50.copy()
            data_corr = self.glcmCorr50.copy()
        
        if row_drop != None:
            data_diss = data_diss.iloc[row_drop:-row_drop, :]
            data_corr = data_corr.iloc[row_drop:-row_drop, :]
        if col_drop != None:
            data_diss = data_diss.iloc[:, col_drop:-col_drop]
            data_corr = data_corr.iloc[:, col_drop:-col_drop]
        plt.figure(figsize=(15,15),frameon=True)
        plt.subplot(1,2,1)
        plt.imshow(np.array(data_diss).astype('float'), cmap='binary', 
                           aspect=None, vmin = diss_min, vmax = diss_max)
        plt.title('Dissimilarity')
        plt.subplot(1,2,2)
        plt.imshow(np.array(data_corr).astype('float'), cmap='binary', 
                           aspect=None, vmin = 0, vmax = 1)
        plt.title('Correlation')
        plt.show()
        return
    
    
    # ACTUAL IMAGE handling
    def showPic(self, bw_inv=0):
        """Draw figure from array shaped image"""
        if bw_inv == 1:
            return Image.fromarray(cv2.bitwise_not(self.pic_bw))
        else:
            return Image.fromarray(self.pic_bw)


def mapGLCM(pic, grid_size = 50, end_p = 0, end_ratioP = 0.05):
    """Break-up image into n squares ('grid_size') and return arrays of
    dissimilarity and correlation using GLCM (Gray Level Co-occurence Matrix)
        
    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    def getGLCM(img): 
        """Calculate co-occurence matrix for the given image and return 
        dissimalirity and correlation measures"""
        #% Calculate dissimilarity
        glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
        measures = (
            greycoprops(glcm, 'dissimilarity')[0, 0],
            greycoprops(glcm, 'correlation')[0, 0]
            )
        return measures
    
    def im_slice(im, n, direction = 'v', ends = 1, end_ratio = 0.025):
        """Slice image into desired amount of horizontal (sideways) or vertical 
        (up-down) pieces"""
        img = im.copy()
        #Create list for images
        img_list = []
        #VERTICAL CASE  
        if direction in ['v', 'vertical', 'y']:
            #Get ends
            if ends == 1:
                end_px = int(round(end_ratio*np.shape(img)[1]))
                end_0 = img[:,0:end_px]
                end_1 = img[:,-end_px:-1]
                img_list.append(np.hstack([end_0, end_1]))
                #Get middle pieces (or all pieces if ends is disabled)
                #Exclude endpieces
                img = img[:,end_px:-end_px]
            n_size = round(np.shape(img)[1]/n)
            for x in range(n):
                img_list.append(img[:,x*n_size:(x+1)*n_size])
        #HORIZONTAL CASE
        if direction in ['h', 'horizontal', 'x']:
            #Get ends
            if ends == 1:
                end_px = int(round(end_ratio*np.shape(img)[0]))
                end_0 = img[0:end_px,:]
                end_1 = img[-end_px:-1,:]
                img_list.append(np.vstack([end_0, end_1]))
                # Get middle pieces (or all pieces if ends is disabled)
                #Exclude endpieces
                img = img[:,end_px:-end_px]
            n_size = round(np.shape(img)[0]/n)
            for x in range(n):
                img_list.append(img[x*n_size:(x+1)*n_size,:])
        return img_list
    
    try: 
        ## Slice sheet up-down
        slices = im_slice(
            pic, 
            grid_size, 
            ends=end_p, 
            direction = 'y', 
            end_ratio = end_ratioP
            )
        
        ## Analyze GLCMs for each slice 
        # Create empty lists for storing results
        diss = []; corr = [];

        for n in range(0, grid_size):
            # Dice slice n into pieces
            dices = im_slice(slices[n], grid_size, direction = 'x', ends = 0)
            # Calculate GLCM for each dice and convert into DataFrame
            glcm = [getGLCM(dices[x]) for x in range(len(slices))]    
            glcm = pd.DataFrame(glcm, columns=['dissimilarity', 'correlation'])
            
            # Add to dissimilarity and correlation arrays
            diss.append(glcm['dissimilarity'].rename(n))
            corr.append(glcm['correlation'].rename(n))
        return (pd.DataFrame(diss).transpose(), 
                pd.DataFrame(corr).transpose())
    except:
        print('Error creating GLCM-map')
        return([], [])

def create_shPic(pic, filename, calc_glcm = 1, glcm_grid = [10, 20, 30, 40, 50]):
    """Create sheet object with the given properties"""
    sheet = sheetPic()
    # Create mirrors of object data masks
    sheet.pic_bw = pic
    sheet.file = filename
    if calc_glcm == 1:
        # Store image file into object
        assert type(pic) == np.ndarray, "Picture type must be np.ndarray type"
        if 10 in glcm_grid:
            sheet.glcmDiss10, sheet.glcmCorr10 = mapGLCM(sheet.pic_bw, grid_size = 10)
        if 20 in glcm_grid:
            sheet.glcmDiss20, sheet.glcmCorr20 = mapGLCM(sheet.pic_bw, grid_size = 20)
        if 30 in glcm_grid:
            sheet.glcmDiss30, sheet.glcmCorr30 = mapGLCM(sheet.pic_bw, grid_size = 30)
        if 40 in glcm_grid:
            sheet.glcmDiss40, sheet.glcmCorr40 = mapGLCM(sheet.pic_bw, grid_size = 40)
        if 50 in glcm_grid:
            sheet.glcmDiss50, sheet.glcmCorr50 = mapGLCM(sheet.pic_bw, grid_size = 50)
    return sheet

def pickleGLCM_fromSheetImg(sh_pic, sh_filename, savepath = None, savename = 'filename', 
                            ret_dict = 0):
    """Read image and save GLCM-data as pickle"""
    p = create_shPic(sh_pic, sh_filename)
    glcms = {'c10': p.glcmCorr10,
             'c20': p.glcmCorr20,
             'c30': p.glcmCorr30,
             'c40': p.glcmCorr40,
             'c50': p.glcmCorr50,
             'd10': p.glcmDiss10,
             'd20': p.glcmDiss20,
             'd30': p.glcmDiss30, 
             'd40': p.glcmDiss40,
             'd50': p.glcmDiss50,
             }
    # Create savename
    if savename == 'filename':
        if '/' in sh_filename:
            fileToSave = sh_filename.split('/')[-1]
        fileToSave = fileToSave.split('.')[0]+'.pkl'
    else:
        fileToSave = savename
    if savepath == None:
        pd.to_pickle(glcms, fileToSave)
        print('Saved '+fileToSave)
    else:
        pd.to_pickle(glcms, savepath+fileToSave)
        print('Saved '+savepath+fileToSave)
    if ret_dict != 0:
        return glcms
    else:
        return