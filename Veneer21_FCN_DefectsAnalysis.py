
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:32:42 2020

@author: h17163
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import math
from PIL import Image
import matplotlib.image as mpimg
#%% Function definitions

def matcherVeneer21(dlist, plist, gridsUsedP1 = [10, 20, 30, 40, 50],
                    votesNeeded = 'half', gridsUsedP2 = [20, 30, 40]):
    """Match lists of objects using GLCM and then slided GLCM for the remaining
    unmatched items. Returns dataframe of matched items and a list of matching
    ratio"""
    # Set votes needed if not defined as number
    if votesNeeded == 'half':
        votesNeeded = math.ceil(len(gridsUsedP1)/2)
    
    # Seek for pairs with GLCM
    pairs, d_rem, p_rem = findPairGLCM_lists(dlist, 
                            plist, 
                            max_iters = 0, 
                            pdropping = 1, #Function default = 1
                            ret_unmatched = 1, 
                            votesNeeded = votesNeeded, 
                            gridsUsed = gridsUsedP1)
    print('\nUnmatched dry: '+str(len(d_rem))+' and wet: '+str(len(p_rem)))
    # Use slided GLCM for the unmatched items (if needed)
    if len(d_rem) > 1 and len(p_rem) > 1:
        iterated = findPairGLCM_slidedMGLists(d_rem, 
                                              p_rem, 
                                              grids = gridsUsedP2)
        matched = pd.concat([pairs, iterated])
    if len(d_rem) == 1 and len(p_rem) == 1:
        # One pair only
        iterated = pd.DataFrame(
            [d_rem[0].file, 
             p_rem[0].file], 
            index = ['dFile', 'pFile']).transpose()
        matched = pd.concat([pairs, iterated])
    if len(d_rem) == 0 and len(p_rem) == 0:
        # No need for slided matching
        matched = pairs
    # Create result list and set drying file in front
    matched = matched[['dFile'] + [ col for col in matched.columns if col != 'dFile' ]]
    matched.reset_index(inplace = True, drop = True)
    # Save resulting dataframe
    res = matched.copy()
    
    # Calculate matching ratio
    matched['pFile'] = matched['pFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')
    matched['dFile'] = matched['dFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')
    matchRatio = sum(matched['dFile'] == matched['pFile'])/len(matched)
    print('\nMatching ratio: '+str(matchRatio))
    return matchRatio, res

def show_imFile(pth, filename, title = None, bw = None):
    if bw != None:
        im = Image.open(pth+filename)
        im.show()
        return
    img = mpimg.imread(pth+filename)
    imgplot = plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.show()
    return

    
def findPairGLCM_double(dsheet, psheets, acceptThr = 3, voteThr = 2, 
                        rankThr = 1, slideRows = 6, ret_shrink = 1,
                        forceSlide = 0):
    """Compare GLCM and the verify with sliding"""
    # GLCM-pair finding
    a = findPairGLCM_cutRange(dsheet, psheets, minVote=voteThr, 
                              accept_rate=acceptThr, minRank = rankThr,
                              ret_shrink=ret_shrink)
    a = pd.DataFrame(a[0:3], index = ['pFile1', 'gridCount1', 'rank'])
    print(a)
    
    # Calculate slided match
    if str(a[0]['pFile1']) != 'nan' or forceSlide == 1:
        # GLCM-sliding
        b = findPairGLCM_slidedMultiGrid(dsheet, psheets, slideWindow=slideRows)
        b = pd.DataFrame(b, index = ['pFile2', 'cum_diff', 'gridCount2'])
        print('Sliding: '+str(b.loc['pFile2'].item()))
       
        # Summarize
        c = pd.concat([a,b]).transpose()
        # Add drying sheet name
        c['dFile'] = dsheet.file
        
        ## Make decision
        # Both are the same
        if c['pFile1'].item() == c['pFile2'].item():
            c['pDecision'] = c['pFile1'].item()
        else:
            c['pDecision'] = np.nan
        
        # Show decision
        print('\n-> Decided '+str(c['pDecision'].item()))
        return c
    else:
        return a
    
    
def compareGLCM(sheet1, sheet2, grid, row_drop = None, col_drop = None):
    """
    Compare GLCM-presentations of two sheets with each other

    Parameters
    ----------
    sheet1 : Object 'sheetP'
        'sheetP'-object containing GLCM map.
    sheet2 : Object 'sheetP'
        (see above)

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    err = 0
    if grid == 10:
        if len(sheet1.glcmDiss10) == 0 and len(sheet2.glcmDiss10) == 0:
            err = 1
    if grid == 20:
        if len(sheet1.glcmDiss20) == 0 and len(sheet2.glcmDiss20) == 0:
            err = 1
    if grid == 30:
        if len(sheet1.glcmDiss30) == 0 and len(sheet2.glcmDiss30) == 0:
            err = 1
    if grid == 40:
        if len(sheet1.glcmDiss40) == 0 and len(sheet2.glcmDiss40) == 0:
            err = 1
    if grid == 50:
        if len(sheet1.glcmDiss40) == 0 and len(sheet2.glcmDiss40) == 0:
            err = 1
    if err == 1:
        print('No GLCM available')
        return (np.nan, np.nan)
    ## Copy data and index if needed
    if grid == 10:
        data_diss1 = sheet1.glcmDiss10.copy()
        data_diss2 = sheet2.glcmDiss10.copy()
        data_corr1 = sheet1.glcmCorr10.copy()
        data_corr2 = sheet2.glcmCorr10.copy()
    if grid == 20:
        data_diss1 = sheet1.glcmDiss20.copy()
        data_diss2 = sheet2.glcmDiss20.copy()
        data_corr1 = sheet1.glcmCorr20.copy()
        data_corr2 = sheet2.glcmCorr20.copy()
    if grid == 30:
        data_diss1 = sheet1.glcmDiss30.copy()
        data_diss2 = sheet2.glcmDiss30.copy()
        data_corr1 = sheet1.glcmCorr30.copy()
        data_corr2 = sheet2.glcmCorr30.copy()
    if grid == 40:
        data_diss1 = sheet1.glcmDiss40.copy()
        data_diss2 = sheet2.glcmDiss40.copy()
        data_corr1 = sheet1.glcmCorr40.copy()
        data_corr2 = sheet2.glcmCorr40.copy()
    if grid == 50:
        data_diss1 = sheet1.glcmDiss50.copy()
        data_diss2 = sheet2.glcmDiss50.copy()
        data_corr1 = sheet1.glcmCorr50.copy()
        data_corr2 = sheet2.glcmCorr50.copy()
    if row_drop != None:
        data_diss1 = data_diss1.iloc[row_drop:-row_drop, :]
        data_diss2 = data_diss2.iloc[row_drop:-row_drop, :]
        data_corr1 = data_corr1.iloc[row_drop:-row_drop, :]
        data_corr2 = data_corr2.iloc[row_drop:-row_drop, :]
    if col_drop != None:
        data_diss1 = data_diss1.iloc[:, col_drop:-col_drop]
        data_corr1 = data_corr1.iloc[:, col_drop:-col_drop]
        data_diss2 = data_diss2.iloc[:, col_drop:-col_drop]
        data_corr2 = data_corr2.iloc[:, col_drop:-col_drop]
    return (cosSim(data_diss1, data_diss2), 
            cosSim(data_corr1, data_corr2))

def findPairGLCM(dsheet_obj, psheet_lst, cut = None, grids = [10, 20, 30, 40, 50],
                 shortOutput = 1, rankThreshold = 1, ret_unique = 0, min_votes = 3,
                 ret_shrink = 0):
    """
    

    Parameters
    ----------
    psheet_obj : TYPE
        DESCRIPTION.
    dsheet_lst : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Set cut to 'None' in case of zero input
    if cut == 0:
        cut = None
    # Extract peeling sheet names
    p_names = [psheet_lst[n].file for n in range(len(psheet_lst))]
    # res_lst = []; 
    res_lstNEW = [];
    # Empty return (no pair is found)
    if ret_shrink == 1:
        empty = (np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        empty = (np.nan, np.nan, np.nan)
    # Run all stored grids
    for n in grids:
            # Create temporary list to store comparison values
            tmp_lst = []
            for k in range(0, len(psheet_lst)):
                try: 
                    tmp_lst.append(
                        compareGLCM(
                            dsheet_obj, 
                            psheet_lst[k], 
                            grid = n,
                            row_drop = cut,
                            col_drop = cut))
                except:
                    tmp_lst.append((np.nan, np.nan))
            resGrid = pd.DataFrame(tmp_lst, columns = ['diss', 'corr'])

            # Calculate ranking (1 = best)
            resGrid[['dissRank', 'corrRank']] = resGrid.rank(ascending = False)
            if cut == None:
                resGrid['cut'] = 0
            else: 
                resGrid['cut'] = cut
            # Add names to result table
            resGrid['pFile'] = p_names
            resGrid['dFile'] = dsheet_obj.file
            # Add cosine similarity average and rank average
            resGrid['CosSimAvg'] = (resGrid[['diss', 'corr']].mean(axis=1))
            resGrid['RankAvg'] = resGrid[['dissRank', 'corrRank']].mean(axis=1)
            # Sort values ascending
            resGrid.sort_values(['RankAvg'], inplace = True)
            # Add grid ID
            resGrid['grid'] = n
            # Append grid to list
            res_lstNEW.append(resGrid)

    if shortOutput == 0:
        return res_lstNEW
    # Create short list aggregation
    shortlist = res_lstNEW[0]
    for n in range(1,len(res_lstNEW)):
        shortlist = pd.concat([shortlist, res_lstNEW[n]])
    # Filter desired columns as answer using the threshold value
    take_cols = ['cut', 
                 'pFile',
                 'dFile', 
                 'CosSimAvg', 
                 'RankAvg', 
                 'grid']
    shortlist = shortlist.loc[shortlist['RankAvg']<=rankThreshold][take_cols]

    if ret_unique == 1:
        # No entries found
        if shortlist.empty:
            return empty
        if shortlist['pFile'].value_counts()[0] >= min_votes:
            voted = shortlist['pFile'].value_counts().index[0]
            ## Calculate shrinking
            p_idx = p_names.index(voted)
            if ret_shrink == 1:
                shrink_abs = (psheet_lst[p_idx].width-dsheet_obj.width)
                shrink_perc = shrink_abs/psheet_lst[p_idx].width
            # Create output
            output = (voted,
                    len(shortlist.loc[shortlist['pFile']==voted])/len(grids),
                    shortlist.loc[shortlist['pFile']==voted]['RankAvg'].mean(),
                    ) 
            if ret_shrink == 1:
                return output + (shrink_abs, round(shrink_perc,5))
            else:
                return output

        else:
            return empty
    return shortlist
    
def findPairGLCM_cutRange(d_sheetobj, psheets, cuts = [0, 1, 2, 3],
                          gridding = [10, 20, 30, 40, 50], accept_rate = 3, 
                          ret_unique = 1, disp_unique = 1, minRank = 1, 
                          minVote = 3, ret_shrink = 1):
    """
    

    Parameters
    ----------
    d_sheetobj : TYPE
        DESCRIPTION.
    psheets : TYPE
        DESCRIPTION.
    cuts : TYPE, optional
        DESCRIPTION. The default is [0, 1, 2, 3].

    Returns
    -------
    None.

    """
    assert accept_rate <= len(cuts), "Accept rate must be equal or smaller than the number of cuts"
    
    # Empty array in case of no match is found
    empty = (np.nan, np.nan, np.nan, np.nan, np.nan)
    # Seek matches using given cuts
    res = [findPairGLCM(
            d_sheetobj, 
            psheets, 
            cut = cuts[n],
            grids = gridding,
            rankThreshold=minRank,
            min_votes=minVote,
            ret_unique=1, 
            ret_shrink=ret_shrink) for n in range(len(cuts))]

    # Convert matches into DataFrame
    res = pd.DataFrame(res, columns = ['pFile', 'grid_ratio', 'rankAvg'])
    res['cut'] = cuts
    # Get unique entries
    if ret_unique == 1:
        res = res.dropna()
        uniques = list(res['pFile'].unique())
        if len(uniques) == 1:
            # Print accept rate
            print('\nAccept rate: '+str(len(res.loc[res['pFile']==uniques[0]]))+' per '+str(len(cuts)))
            if len(res.loc[res['pFile']==uniques[0]]) >= accept_rate:
                # print('Candidate found: '+str(uniques[0]))
                if disp_unique == 1:
                    print('Paired '+str(d_sheetobj.file)+' with '+str(uniques[0]))
                if ret_shrink ==1:
                    return (uniques[0], 
                            res['grid_ratio'].mean(), 
                            res['rankAvg'].mean(),
                            res['shrink_mm'].mean(),
                            res['shrink_ratio'].mean(),
                            )
                else: 
                    return (uniques[0], 
                            res['grid_ratio'].mean(), 
                            res['rankAvg'].mean(),
                            np.nan,
                            np.nan,)
            else:
                print('Acceptance rate too low for pairing')
                return empty
        else:
            return empty
    else:
        return res

def findPairGLCM_lists(dlist, plist, pdropping = 1, duplicateRemoval = 1,
                       iterate = 1, return_df = 1, ret_unmatched = 0, rankGradient = 0.5, 
                       votesNeeded = 3, max_iters = 100, gridsUsed = [10, 20, 30, 40, 50],
                       cutsUsed = [1,2,3], ret_shrink = 0):
    """
    OBSERVE: If list of peeling objects is given first the shrinkage will be
    given as negative numbers

    Parameters
    ----------
    plist : TYPE
        DESCRIPTION.
    dlist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert type(votesNeeded)==int, "Provide 'votesNeeded' as an integer"
    # Create iteration list
    iter_lst = []
    # Set starting rank average as 1 (only best matches)
    rnk_avg = 1
    # Start iteration count
    i_count = -1
    while len(plist) != 0 or len(dlist) != 0:
        # Save iteration count
        i_count = i_count + 1
        print('\nStarting iteration: '+str(i_count))
        # Take drying sheet names as a list
        d_names = [dlist[n].file for n in range(len(dlist))]
        # Create empty lists for results and dropped peeling entries
        lst = []; pdrops = []; 
        # Copy the original list of peeling sheets
        p_objsTmp = plist.copy()
        for n in range(len(dlist)):
            lst.append(findPairGLCM_cutRange(
                    dlist[n], 
                    p_objsTmp, 
                    cuts = cutsUsed,
                    gridding = gridsUsed,
                    ret_unique=1, 
                    accept_rate=1,
                    minRank = rnk_avg,
                    minVote = votesNeeded,
                    ret_shrink = ret_shrink,
                    )
                )
            if pdropping == 1:
                pdrops.append(lst[-1][0])
                # Clear temporary list of peeling sheets
                p_objsTmp = []
                # Update temporary list 
                for x in range(0, len(plist)):
                    # Filename is not in the drop list...
                    if plist[x].file not in pdrops:
                        # Add filename to temporary list
                        p_objsTmp.append(plist[x])
        
        # Create DataFrame. Add drying sheet names and iteration round number
        df_lst = pd.DataFrame(lst, columns = ['pFile', 
                                              'gridCount', 
                                              'rank',
                                              'shrink_mm',
                                              'shrink_ratio',
                                              ])
        df_lst['dFile'] = d_names
        df_lst['iteration'] = i_count
        print(df_lst)
        # Remove dupllicates
        if duplicateRemoval == 1:
            duplicates = df_lst.dropna().loc[df_lst['pFile'].duplicated(keep=False)]
            df_lst.replace(list(duplicates['pFile']), np.nan, inplace=True)
            print('\n'+str(len(duplicates))+' duplicates removed')
        
        # Calculate number of matches
        match_count = len(df_lst['pFile'].dropna())-len(duplicates)
        print('Number of matches found: '+str(match_count))
        
        # Alter matching conditions if no matches were found
        if match_count == 0:
            # SECONDARY MEASURE: Lower rank average requirement
            if votesNeeded == 1:
                rnk_avg = rnk_avg + rankGradient
                print('-> Relaxed rank average requirement to: '+str(rnk_avg))
            # PRIMARY MEASURE: Lower voting threshold (min = 1)
            if votesNeeded != 1:
                votesNeeded = votesNeeded - 1
                print('-> Relaxed minimum vote requirement to: '+str(votesNeeded))
            
        # Return answer if no iteration is desired
        if iterate == 0:
            break
        
        # Otherwise append result to iterated results
        iter_lst.append(df_lst.dropna(subset = ['pFile']))
        
        #Create drop lists for peeling and drying sheets
        p_drop_list = list(df_lst.dropna(subset = ['pFile'])['pFile'])
        d_drop_list = list(df_lst.dropna(subset = ['pFile'])['dFile'])
        
        #Create new lists by dropping extra entries
        # Peeling
        plist_iter = []
        for x in range(0, len(plist)):
            # If file is not in drop list then add to iteration list
            if plist[x].file not in p_drop_list:
                plist_iter.append(plist[x])
        # Drying
        dlist_iter = []
        for x in range(0, len(dlist)):
            if dlist[x].file not in d_drop_list:
                dlist_iter.append(dlist[x])

        # Replace plist and dlist with iterated lists
        plist = plist_iter.copy()
        dlist = dlist_iter.copy()
        
        # Check if iteration limit is reached
        if type(max_iters) == int:
            if i_count >= max_iters:
                print('\nMaximum iteration rounds reached (n = '+str(max_iters)+')')
                break
        
    ## WHILE-loop ending
    if iterate == 0:
        return df_lst
    print('Iteration completed')
    if return_df == 1:
        res = pd.DataFrame()
        for n in range(len(iter_lst)):
            res = pd.concat([res, iter_lst[n]]).reset_index(drop=True)
        # Return unmatched for further iteration?
        if ret_unmatched==1:
            return (res, dlist, plist)
        else:
            return res
    return iter_lst    

def cosSim(ser1, ser2):
    """Calculate cosine similarity from two pandas series"""
    a = np.array(ser1).reshape(1,-1)
    b = np.array(ser2).reshape(1,-1)
    return cosine_similarity(a,b).item()


def subset_dict(dct, start, stop, retIndex=0, resetIndex=1):
    """Take subset of a large dictionary and reset keys-index default"""    
    dct = dict(list(dct.items())[start:stop])
    if retIndex==1:
        return [list(dct.items())[n][0] for n in range(len(dct))]
    if resetIndex==1:
        return dict((i,dct[k]) for i,k in enumerate(sorted(dct.keys())))
    else:
        return dct

def findPairGLCM_slided(dsheet_obj, psheet_objs, grid, slideWindow = 6, 
                        retShort = 1, printOut = 0):
    """
    

    Parameters
    ----------
    dsheet_obj : TYPE
        DESCRIPTION.
    psheet_objs : TYPE
        DESCRIPTION.
    grid : TYPE
        DESCRIPTION.
    slideWindow : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None.

    """
    def fetchGLCM(sheet_obj, grid):
        """Take desired grid from sheet object. Returns tuple where index 0
        is correlation and index 1 is dissimilarity"""
        if grid == 10:
            corr = sheet_obj.glcmCorr10
            diss = sheet_obj.glcmDiss10
        elif grid == 20:
            corr = sheet_obj.glcmCorr20
            diss = sheet_obj.glcmDiss20
        elif grid == 30:
            corr = sheet_obj.glcmCorr30
            diss = sheet_obj.glcmDiss30
        elif grid == 40:
            corr = sheet_obj.glcmCorr40
            diss = sheet_obj.glcmDiss40
        elif grid == 50:
            corr = sheet_obj.glcmCorr50
            diss = sheet_obj.glcmDiss50
        else:
            print('No grid '+str(grid)+' found')
            return(np.nan, np.nan)
        return (corr, diss)

    ## Take dryer glcms
    dCorr, dDiss = fetchGLCM(dsheet_obj, grid)
    
    ## Create dryer sheet slide
    # Even or odd number
    if slideWindow % 2 == 0:
        dCorr_piece = dCorr[int(slideWindow/2):-int(slideWindow/2)]
        dDiss_piece = dDiss[int(slideWindow/2):-int(slideWindow/2)]
    else:
        print('Provide an even number for sliding window')
        return
    
    ## Match slide with peeling sheets
    # Sliding index from zero to desired slide window
    sliding = list(range(0, slideWindow+1))
    
    cFinal = []
    dFinal = []
    # cFinal = pd.DataFrame()
    # dFinal = pd.DataFrame()
    
    for k in range(0, len(psheet_objs)):
        # print(k)
        ## Create peeling sheet slides
        p = psheet_objs[k]
        ## Take peeling glcms
        pCorr, pDiss = fetchGLCM(p, grid)
        
        # Cut indices FROM sliding start index TO end index
        cut_list = [
                    [sliding[n], 
                     len(dCorr)-sliding[-1-n]
                     ] for n in sliding] 
        
        #Store pieces of dryer sheet correlations
        pCorr_pieces = [pCorr[cut_list[n][0]:cut_list[n][1]] for n in range(len(cut_list))]
        pDiss_pieces = [pDiss[cut_list[n][0]:cut_list[n][1]] for n in range(len(cut_list))]
        
        corrs = []
        dissims = []
        for m in range(len(pCorr_pieces)):
            try:
                corrs.append(cosSim(dCorr_piece, pCorr_pieces[m]))
                dissims.append(cosSim(dDiss_piece, pDiss_pieces[m]))
            except:
                corrs.append(np.nan)
                dissims.append(np.nan)
        # Add sheet name
        corrs.append(p.file)
        dissims.append(p.file)
    
        # Add results to final listings
        cFinal.append(corrs)
        dFinal.append(dissims)
    
    # Create DataFrames
    cFinal = pd.DataFrame(cFinal)
    dFinal = pd.DataFrame(dFinal)
    
    # Add max-values
    cFinal['max'] = cFinal.iloc[:,:-1].max(1)
    dFinal['max'] = dFinal.iloc[:,:-1].max(1)
    
    # Sort by maximum and reset index
    cFinal.sort_values('max', ascending = False, inplace = True)
    dFinal.sort_values('max', ascending = False, inplace = True)
    
    cFinal.reset_index(inplace = True)
    dFinal.reset_index(inplace = True)
    
    # print('Corr:\t '+str(cFinal.iloc[0,-2])+'\nDiss:\t '+str(dFinal.iloc[0,-2]))
    
    if retShort == 1:
        # Get best, similarity and difference to second best
        cSumm = (cFinal.iloc[0, -2], round(cFinal['max'][0], 4), round(cFinal['max'][0]-cFinal['max'][1], 4), grid, 'corr')
        dSumm = (dFinal.iloc[0, -2], round(dFinal['max'][0], 4), round(dFinal['max'][0]-dFinal['max'][1], 4), grid, 'diss')
        summ = pd.DataFrame([cSumm, dSumm])
        summ.columns = ['best', 'simil', 'diffToSec', 'grid', 'measure']
        if printOut == 1:
            print(summ)
        return(summ)
    
    return(cFinal, dFinal)

def findPairGLCM_slidedMultiGrid(dsheet_obj, psheet_objs, grids = [20, 30, 40], 
                                 slideWindow = 6, sorting = 'diff', shortOut = 1, 
                                 printOut = 0):
    """
    Slide multiple grids and return the most probable match. Sorting method: 
        'diff': sum of differences for match between best and second (default)
        'count': number of match occurences in results

    """
    # Similarities
    sims = [findPairGLCM_slided(dsheet_obj, psheet_objs, x, slideWindow,
                                retShort = 1, printOut = printOut) for x in grids]
    # Create one dataframe for results
    result = pd.DataFrame()
    for x in range(len(sims)):
        result = pd.concat([result, sims[x]])
    # Reset index
    result.reset_index(drop=True, inplace=True)
    
    # Pivot result
    res_pivot = result.pivot_table('diffToSec', 'best', aggfunc=[np.sum, len], fill_value=0)
    
    # Sort values based on user preference
    if sorting == 'diff':
        res_pivot.sort_values(('sum', 'diffToSec'), ascending = False, inplace = True)
    if sorting == 'count':
        res_pivot.sort_values(('len', 'diffToSec'), ascending = False, inplace = True)
    
    # Short output
    if shortOut == 1:
        return [res_pivot.index[0], 
                res_pivot[('sum', 'diffToSec')][0],
                res_pivot[('len', 'diffToSec')][0]/(len(grids)*2)]
    
    return res_pivot

def findPairGLCM_slidedMGLists(dsheet_objs, psheet_objs, iRounds = 10, 
                               grids = [20, 30, 40]):
    """Match lists using slided GLCM with multiple grids"""
    # Set iteration counter to zero
    iround = 0
    # Create DataFrame of accepted sheets
    iset_ok = pd.DataFrame()
    
    while iround < iRounds:
        print('Starting iteration '+str(iround))
        print(len(dsheet_objs))
        print(len(psheet_objs))
        # Get filenames
        dfiles = [dsheet_objs[n].file for n in range(len(dsheet_objs))]
        
        #Iterate remaining sheets
        iset = []
        for x in range(len(dsheet_objs)):
            print(dfiles[x])
            pair = findPairGLCM_slidedMultiGrid(dsheet_objs[x], 
                                                psheet_objs,
                                                grids = grids)
            print(pair)
            iset.append(pair+[dfiles[x]])
            
        iset = pd.DataFrame(iset)
        iset.columns = ['pFile', 'diff', 'ratio', 'dFile']
        iset.sort_values(['pFile', 'diff', 'ratio'], ascending=[True, False, False], inplace = True)
        iset_ok = pd.concat([iset_ok, iset.drop_duplicates(subset='pFile')])
        
        ## Create new lists by dropping matched entries
        p_drop_list = list(iset_ok['pFile'])
        d_drop_list = list(iset_ok['dFile'])
        
        # Peeling
        plist_iter = []
        for x in range(0, len(psheet_objs)):
            # If file is not in drop list then add to iteration list
            if psheet_objs[x].file not in p_drop_list:
                plist_iter.append(psheet_objs[x])
        # Drying
        dlist_iter = []
        for x in range(0, len(dsheet_objs)):
            if dsheet_objs[x].file not in d_drop_list:
                dlist_iter.append(dsheet_objs[x])
        
        # Check if only one entry left in both lists
        if len(plist_iter) == 1 and len(dlist_iter) == 1:
            last = pd.DataFrame([plist_iter[0].file, 
                                 dlist_iter[0].file], 
                                 index=['pFile', 'dFile']).transpose()
            iset_ok = pd.concat([iset_ok, last])
            print('No sheets left to match')
            break
        # Break loop if no entries left
        if len(plist_iter) == 0 or len(dlist_iter) == 0:
            print('No sheets left to match')
            break
                
        # Start loop over again
        dsheet_objs = dlist_iter
        psheet_objs = plist_iter
        
        # Add iteration count
        iround = iround + 1
    
    return iset_ok

def getGLCM(img): 
    """Calculate co-occurence matrix for the given image and return 
    dissimalirity and correlation measures"""
    #% Calculate dissimilarity
    if np.shape(img)[0] == 0 or np.shape(img)[1] == 0:
        print('Empty image data')
        return (0,0)
    glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    measures = (
        greycoprops(glcm, 'dissimilarity')[0, 0],
        greycoprops(glcm, 'correlation')[0, 0]
        )
    return measures

def getGLCM_Multi(img, sz = 100, px = range(1,50), outRawGLCM = 0,
                  ang = [0, np.pi/4, np.pi/2, np.pi*3/4],
                  prop = 'contrast', df_out = 0):
    """Calculate graycoproperties for a miniaturized image"""
    # Resize image
    imgSmall = resize_img(img, sz)
    
    # Calculate GLCM-matrices
    glcm = greycomatrix(imgSmall, px, ang)
    
    # Return GLCM-matrix?
    if outRawGLCM != 0:
        return glcm
    
    # Extract property of interest
    p = greycoprops(glcm, prop)
    
    # Return answer as DataFrame
    if df_out == 1:
        return pd.DataFrame(p)
    else: 
        return p

def resize_img(image, px_size = 400):
        """Resize image stored as numpy array. From\
        https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv"""
        r = px_size / image.shape[1]
        dim = (px_size, int(image.shape[0] * r))
        # perform the actual resizing of the image and show it
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

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

def im_blocks9(im, nr):
    """Return part of image divided into nine equally-sized blocks"""
    # Get dimensions and calculate width and height of one block
    assert nr in list(np.arange(1,9+1)), "Provide number between 1-9"
    dims = np.shape(im)
    x_size = dims[0]//3
    y_size = dims[1]//3
    block_nr = {1: (0,0), 2: (0,1), 3: (0,2), 4: (1,0), 5: (1,1), 6: (1,2),
                7: (2,0), 8: (2,1), 9: (2,2),}
    # Calculate coordinates to take
    x0 = block_nr[nr][0]*x_size
    x1 = (block_nr[nr][0]+1)*x_size
    y0 = block_nr[nr][1]*y_size
    y1 = (block_nr[nr][1]+1)*y_size
    return im[x0:x1, y0:y1]

def show_im(image, bw_inv=0):
        """Draw figure from array shaped image"""
        if bw_inv == 1:
            return Image.fromarray(cv2.bitwise_not(image))
        else:
            return Image.fromarray(image)

def read_rotate_resize(filePath, size=800, cropping = 1, top_cuts = 0, 
                       side_cuts = 0, rot_rng = [-10, 10], rot_decim = 4, bw=1,
                       crop_thresh = 255):
    """Read raw image file to numpy array and resize into desired size"""
    
    def rotate_image(image, angle):
        """Rotate image using given angle"""
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, 
                              borderValue=(255,255,255))
        return result
    
    def resize_img(image, px_size = 400):
        """Resize image stored as numpy array. From\
        https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv"""
        r = px_size / image.shape[1]
        dim = (px_size, int(image.shape[0] * r))
        # perform the actual resizing of the image and show it
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    def crop_image(image, pixel_value=255):
        """Crop image (whitespace excluded)"""
        crop_rows = image[~np.all(image == pixel_value, axis=1), :]
        cropped_image = crop_rows[:, ~np.all(crop_rows >= pixel_value, axis=0)]
        return cropped_image
    #Read image in black-white, leave out whitespace and return resized
    img = cv2.imread(filePath,0)
    #Crop out whitespaces from the original image
    if cropping == 1:
        img = crop_image(img[top_cuts:-1-top_cuts,side_cuts:-1-side_cuts], pixel_value=crop_thresh)
    
    #Rotate cropped image
    angles = []
    angles.append(list(np.arange(rot_rng[0], rot_rng[1]+1, 1)))
    if rot_decim > 0:
        angles.append(list(np.arange(-9,10, 1)/10))
    if rot_decim > 1:
        angles.append(list(np.arange(-9,10, 1)/100))
    if rot_decim > 2:
        angles.append(list(np.arange(-9,10, 1)/1000))
    if rot_decim > 3:
        angles.append(list(np.arange(-9,10, 1)/10000))

    #Calculate mean whiteness for rotation using small image
    img_small = resize_img(img, px_size=size)
    
    #Rotate to desired decimals
    for x in range(len(angles)):
        if x == 0:
            rot0 = 0
        else:
            rot0 = rot_opt
        rots = [np.mean(crop_image(rotate_image(img_small, rot0+angles[x][n]))) 
                for n in range(len(angles[x]))]
        rots = pd.DataFrame([angles[x], rots], index=['angle', 'bright']).transpose()
        rots['angle'] = rots['angle']+rot0
        # Get rotation angle
        try:
            rot_opt = round(rots.loc[rots['bright'] == min(rots['bright'])]['angle'],5).item()
        except:
            pass
    # Set final rotation angle
    rot_1 = rot_opt
    print('Rotated '+str(round(rot_1, rot_decim)))
    if rot_1!=0:
        #Rotate full sized image
        img = crop_image(rotate_image(img, rot_1))
    if bw == 0:
        return resize_img(cv2.imread(filePath), px_size = size)
    else:
        return crop_image(resize_img(img, px_size = size))

def shrink_img(img_to_shrink, small):
    """Scales down large image to the size of small"""
    # Calculate decrease ratios
    x_decr = small.shape[0]/img_to_shrink.shape[0]
    y_decr = small.shape[1]/img_to_shrink.shape[1]
    
    # Convert to pixels
    x_pix = int(img_to_shrink.shape[0] * x_decr)
    y_pix = int(img_to_shrink.shape[1] * y_decr)
    
    out = cv2.resize(img_to_shrink, (y_pix, x_pix))
    return out

def im_dice(im, xdir, ydir, end_ratioP = 0.005):
    
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

    ## Slice sheet up-down
    slices = im_slice(
        im, 
        ydir, 
        direction = 'y', 
        end_ratio = end_ratioP, 
        ends = 0,
        )
    
    dices = []
    for n in range(0, xdir):
        # Dice slices n into pieces
        dices.append(im_slice(slices[n], xdir, direction = 'x', ends = 0))
    # Convert to 1D-list
    dices_final = []
    for m in range(0, len(dices)):
        for k in range(0, len(dices[m])):
            dices_final.append(dices[m][k])
    return dices_final
