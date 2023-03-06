#%% Import libraries
import pandas as pd
import numpy as np
import os

#%% Settings
# This is the result produced in matching phase ("...READ_and_MATCH_images.py")
matchingResultsFolder = '.../Matched/'

#%% Read example file
def checkAcc(df_list):
    """Check matching accuracy"""
    for m in range(len(df_list)):
        matched = df_list[m]
        matched['pCheck'] = matched['pFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')
        matched['dCheck'] = matched['dFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')

    acc = []
    for x in range(len(df_list)):
        acc.append(sum(df_list[x]['pCheck'] == df_list[x]['dCheck'])/len(df_list[x]))
    
    print()
    return (np.mean(acc), np.std(acc))


#%% Check accuracies
# Read Files
matchFiles = os.listdir(matchFolder)

# Accuracies
accs = []
for x in range(len(matchFiles)):
    mdata = pd.read_pickle(matchFolder+matchFiles[x])
    accs.append(checkAcc(mdata))
accs = pd.DataFrame(accs, columns = ['acc', 'std'])
accs['summaryFile'] = matchFiles
accs['gridCount'] = accs['summaryFile'].str.split('_').apply(pd.Series)[1]

# Inspect gridding
gridding = accs['summaryFile'].str.split('_').apply(pd.Series).iloc[:,3:]
# Reset column index
gridding.columns = range(gridding.columns.size)
for x in range(len(gridding.columns)):
    gridding[x] = gridding[x].str.replace('.pkl','').str.replace(',','')
# Convert to integer
gridding = gridding.fillna(0).astype(int)
accs = pd.concat([accs, gridding], axis=1)

#%% ANALYZE UNMATCHED TO SPOT REGULARLY OCCURING WRONG PAIRS
# (Often wrongly labeled pairing...)

unmatchedFull = pd.DataFrame()
for x in range(len(matchFiles)):
    mdata = pd.read_pickle(matchFolder+matchFiles[x])
    # Extract sampleSize
    try:
        # Original in use
        sampleSize = int(matchFiles[x].split('Data')[1].split('_n')[0])
    except:
        # Clean in use
        sampleSize = int(matchFiles[x].split('Clean')[1].split('_n')[0])
    
    for m in range(len(mdata)):
        matched = mdata[m]
        matched['pCheck'] = matched['pFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')
        matched['dCheck'] = matched['dFile'].str.replace('.png','').str.replace('wet_','').str.replace('dry_','')
        unmatched = matched.loc[matched['pCheck'] != matched['dCheck']]
        if not unmatched.empty:
            unmatched['sampleSize'] = sampleSize
            unmatched['sampleIter'] = m
            unmatchedFull = pd.concat([unmatchedFull, unmatched])
# print(matched.loc[matched['pCheck'] != matched['dCheck']])
unmatchedFull.reset_index(drop=True, inplace=True)
print('\nSheets with the highest number of matching errors')
print(unmatchedFull['dCheck'].value_counts())
