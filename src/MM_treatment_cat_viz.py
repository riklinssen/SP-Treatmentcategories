

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.cm
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import glob



###visualisations SP graphs. 
###repetated cross-section matched target group baseline endline (target group only)
### Rik Linssen - Jan 2020
### reads in xls with estimated means, se, ATT's and se as a result of the analyses in stata --> for each treatment category --> src\data\results_graphs_treatment_category_1.csv
### reads in xls with variable and val labels for each outcome variable.  --> ValueLabels.xls -src\data\Value labels_mya_2.xlsx
### reads in xls with graph labels (item/question texts per outcome variable) --> src\data\VariableLabels_mya_2.xlsx
# 1 create set with plot titles labels and limits =plotlabels

# 2 create set with actual data 

# in plotting functions make sure data refers to correct plotsettings set


#################Filepaths

##enter your filepaths here. 
#cwd means your current working directory.
# change outfile directories relative to directory of this script. 

base_path = Path.cwd()
data_path=base_path/"data"
graphs_path=base_path/"graphs"

csvfiles = glob.glob("data\*.csv")
dflist=[]
for f in csvfiles: 
    name=f[-14:-4]
    df=((pd.read_csv(f, sep=';', decimal=',', header=0))
    .rename(columns=str.lower)
    .assign(category_nr=lambda x: name)
    .assign(tr_cat=lambda x: x['category_nr'].map({
        'category_1': 'none', 
        'category_2': 'only awareness',
        'category_3': 'awareness + training',
        'category_4': 'awareness + training + advocacy'
        })) 
         )
    dflist.append(df)


results=pd.concat(dflist, ignore_index=True)

vallabelfilename=data_path/"Value labels_mya_2.xlsx"
varlabelfilename=data_path/"VariableLabels_mya_2.xlsx"

#vallabs and varlabs
vallabs=pd.read_excel(vallabelfilename, header=None, names=['labelname', 'value', 'valuelabel'])#, names
varlabs=pd.read_excel(varlabelfilename, usecols=['name', 'vallab', 'varlab'])

#create dataframe with relevant labels, limits etc for the visualisations = plotlabels

#rename varlabs or ease referencings later
varlabs.columns=['name', 'vallab', 'titles']


##vallabs cleanup
#set 88 dk & 99 refuse to nan. new col sel value_m 
missingvals=[88,99]
vallabs['value_m']=vallabs['value'].apply(lambda x: x if x not in missingvals else np.nan)

#  need the labelname, 'ymin', 'yminl',  'ymax', 'ymaxl'

labelnames=vallabs.labelname.unique()

labelset=[]
#telkens een rij er bij 
for label in labelnames:
    #select relevant rows in vallabs df
    sel=vallabs.loc[vallabs['labelname']==label]
    #get index of min and max values (in case these are not sorted)
    low=sel['value_m'].idxmin()
    high=sel['value_m'].idxmax()
    #then select the value and the label
    yminv=sel.loc[low,'value_m']
    yminl=sel.loc[low,'valuelabel']
    ymaxv=sel.loc[high, 'value_m']
    ymaxl=sel.loc[high,'valuelabel']
    #add the varname as wel
    labelset.append([label, yminv, yminl, ymaxv, ymaxl])    
#add a cell with all values and labels. 
vallabelset=pd.DataFrame(labelset, columns=['vallab', 'yminv', 'yminl', 'ymaxv', 'ymaxl'])

# now switch the int in the yminl & ymmaxl so that the nrs appear close to the y-axis

def labelswitch(x):
    oldval=x
    cut=oldval[:1]
    newval='-'+ oldval[3:]+ '- '+ cut
    return newval


#switch labels make sure max and min vals are ints
for c in ['yminl', 'ymaxl']: 
    vallabelset[c]=vallabelset[c].apply(labelswitch)

# max values should be ints

for c in ['yminv', 'ymaxv']: 
    vallabelset[c]=vallabelset[c].astype(int)




#link to variables


plotlabels=pd.merge(left=varlabs, right=vallabelset, left_on=['vallab'], right_on=['vallab'] )






colnamemapper={'name' : 'Variable', 'group' : 'Group', 'mean' : 'Mean' , 'se' : 'SE', 'ub': 'CI_lowerbound', 'lb' :'CI_upperbound', 'sample': 'n' } 
results=results.rename(columns=colnamemapper)


plotlabels_r=results.loc[:,['Variable', 'Group']]

#all columns lowercase
plotlabels_r.columns=map(str.lower, plotlabels_r.columns)

#new varname, stip and split each element 
#marieke's stuff can be sliced at position 2 use: 
#plotlabels_r['varname']=[x.strip()[2:] for x in  plotlabels_r['resultvar']] 
#check at which position string needs to be sliced. 
#make new col with resultvar(to avoid confusion across dfs)
#trim spaces in resultvar & variable

plotlabels_r['resultvar']=plotlabels_r['variable'].apply(lambda x: x.strip())
plotlabels_r=plotlabels_r.drop(columns=['variable'])

resultnames=list(plotlabels_r['resultvar'].unique())
originalnames=list(plotlabels['name'])


def whichoriginalvar(x):
       
    """Finds originalname used in datacleaning for resultvar in resultssets
       
    :param x: string with resultvar
    :type x: str
    :return: substring from originalvar contained in resultvar
    :rtype: str
    """ 
    for name in originalnames: 
        if x.endswith(name):
            return name


plotlabels_r['ovarname']=plotlabels_r['resultvar'].apply(whichoriginalvar)




##see if there are missing names/unlabeled variables


resultvars = set(plotlabels_r['ovarname'])
originalvars = set(plotlabels['name'])
#check if there is anything missing in the varlabs df 
missing_resultvarlabels=list(set(resultvars) - set(originalvars))
print(missing_resultvarlabels)

#if len(missing_resultvarlabels)>0: 
#    missing_resultvarlabels




#Add the correct labels to plotlabels_r

plotlabels_figs=pd.merge(left=plotlabels_r, right=plotlabels, how='left', left_on='ovarname', right_on='name')

# now I have 2 lines for each label
plotlabels_figs=plotlabels_figs.drop_duplicates(subset=['resultvar'])

#drop irrelevant info use loc 
plotlabels_figs=plotlabels_figs[['resultvar', 'ovarname', 'name', 'vallab', 'titles',
       'yminv', 'yminl', 'ymaxv', 'ymaxl']]




#make plotlabels in dict for easy referencing. key is resultvar
plotlabels_f_dict=plotlabels_figs.set_index('resultvar').to_dict(orient='index')



###############
## try plots. 

resultdata=results.copy()

resultdata['err']=resultdata['CI_upperbound']-resultdata['Mean']


##Now load in data tovisualise
data=resultdata.drop(columns=['CI_upperbound', 'CI_lowerbound'])


#coloring
#color all endline and difference bars per treatment cat, baseline always green. 
cmap = {'category_1':  '#630235',
        'category_2':  '#53297D',
        'category_3':  '#0B9CDA',
        'category_4': '#F16E22'}

data['color']=data['category_nr'].map(cmap)

data['color']=np.where((data.Group=='Baseline'), '#44841A', data.color)

# select those variables which are scales, those which are percentages. 
# only one var is a percentage: out_voice_alt

data['ispercentage']='no'
data['ispercentage']=np.where((data.Variable=='out_voice_alt'), 'yes', data.ispercentage)


###Plot loop for scale vars 

#make lists to loop over
#make a set for the data that is in some kind of scale
data_scales=data.loc[data['ispercentage']=='no']

scalevariablelist=data_scales['Variable'].unique()
# make a set for the data that is a proportion
data_prop=data.loc[data['ispercentage']=='yes']
propvariablelist=data_prop['Variable'].unique()



