

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
from matplotlib.gridspec import GridSpec
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

countryname='Source: SP VNM R2F surveys, '

base_path = Path.cwd()
data_path=base_path/"data/vnm"
graphs_path=base_path/"graphs"
alt_graphs_path=base_path/"alt_graphs" #for alternative graphs

csvfiles = glob.glob("data/vnm/*.csv")
dflist=[]
for f in csvfiles: 
    name=f[-14:]
    print(name)

    df=((pd.read_csv(f, sep=';', decimal=',', header=0))
    .rename(columns=str.lower)
    .assign(category_nr=lambda x: name)
    .assign(tr_cat=lambda x: x['category_nr'].map({
        'category_1.csv': 'None', 
        'category_2.csv': 'Communication + Advocacy',
        'category_3.csv': 'Communication + Advocacy + Monitoring',
        'category_4.csv': 'Communication + Advocacy + Monitoring + Training'
        })) 
         )
    dflist.append(df)


results=pd.concat(dflist, ignore_index=True)

vallabelfilename=data_path/"Value labels_vie.xlsx"
varlabelfilename=data_path/"VariableLabels_vie.xlsx"

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
results['err']=results['CI_upperbound']-results['Mean']

#avg=results.loc[results.Group.isin(['Baseline', 'Endline'])]
#dif=results.loc[results.Group =='Difference']




##Now load in data tovisualise
data=results.drop(columns=['CI_upperbound', 'CI_lowerbound'])


#coloring
#color all endline and difference bars per treatment cat, baseline always green. 
cmap = {'category_1.csv':  '#630235',
        'category_2.csv':  '#53297D',
        'category_3.csv':  '#0B9CDA',
        'category_4.csv': '#F16E22'}

data['color']=data['category_nr'].map(cmap)

data['color']=np.where((data.Group=='Baseline'), '#44841A', data.color)

# select those variables which are scales, those which are percentages. 
# only one var is a percentage: out_voice_alt


data['ispercentage']='no'
data['ispercentage']=np.where((data.Variable.isin(['out_f4dtakenactioncso', 'out_f4dtakenactiontax', 'out_f4dstat_activepassive'])), 'yes', data.ispercentage)


###Plot loop for scale vars 

#make lists to loop over
#make a set for the data that is in some kind of scale
data_scales=data.loc[data['ispercentage']=='no']

# make a set for the data that is a proportion
data_prop=data.loc[data['ispercentage']=='yes']
data_scales=data.loc[data['ispercentage']=='no']

#take a set with differences only

#make indexslice 
idx = pd.IndexSlice
difference_scales=data_scales.loc[data_scales['Group']=='Difference'].set_index(['Variable', 'tr_cat', 'Group']).sort_index()
means_scales=data_scales.loc[data_scales['Group'].isin(['Baseline', 'Endline'])].set_index(['Variable', 'tr_cat', 'Group']).sort_index()

difference_props=data_prop.loc[data_prop['Group']=='Difference'].set_index(['Variable', 'tr_cat', 'Group']).sort_index()
means_props=data_prop.loc[data_prop['Group'].isin(['Baseline', 'Endline'])].set_index(['Variable', 'tr_cat', 'Group']).sort_index()


#make a list with scale variables

scalevariablelist=list(means_scales.index.get_level_values(0).unique())
propvariablelist=list(means_props.index.get_level_values(0).unique())

#make a list with treatment categories (also for sorting)
catlist=['None', 'Communication + Advocacy', 'Communication + Advocacy + Monitoring', 'Communication + Advocacy + Monitoring + Training']

#make a new colormap with category labels
cmaplabel=dict(zip(catlist, list(cmap.values())))



def autolabel(bars, xpos='center'):
    """
    Attach a text label above each bar in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        height = round(bar.get_height(),1)
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom', color=bar.get_edgecolor(), alpha=1)




def autolabeldf(bars, xpos='center'):
    """
    Attach a text label above each difference bar in *ax*, displaying its height for difference bars (includes bottom)

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        value=round(bar.get_height(),1)
        height = bar.get_height() + bar.get_y()
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], 1.01*height,
                '{}'.format(value), ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)


def autolabelpercent(bars, xpos='center'):
    """
    Attach a text label above each bar in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        height = bar.get_height()
        heightp=str(int(bar.get_height()*100))+ '%'
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], 1.03*height,
                heightp, ha=ha[xpos], va='bottom', color=bar.get_edgecolor(), alpha=1)




def autolabelpercentmid(bars, xpos='center'):
    """
    Attach a text label above each bar in *ax*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        height = bar.get_height()
        heightp=str(int(bar.get_height()*100))+ '%'
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], height/2,
                heightp, ha=ha[xpos], va='bottom', color='black')




def autolabeldfpercent(bars, xpos='center'):
    """
    Attach a text label above each difference bar in *ax*, displaying its height for difference bars (includes bottom)

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        value=str(int(round(bar.get_height()*100,1)))+ '%'
        height = bar.get_height() + bar.get_y()
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], 1.01*height,
                '{}'.format(value), ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)


def autolabeldfpercenth(bars, xpos='center'):
    """
    Attach a text label for each difference horizontal bar in *ax*, displaying its height for difference bars (includes bottom)

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        value=str(int(round(bar.get_width()*100,1)))+ '%'
        width = bar.get_height() + bar.get_y()
        axs[i].text(bar.get_x() + bar.get_width()*offset[xpos], 1.01*width,
                '{}'.format(value), ha=ha[xpos], va='bottom', color=bar.get_facecolor(), alpha=1)
            


#add bottom to difference scales=baseline value from means_scales
difference_scales['bottom']=means_scales.loc[idx[:,:,'Baseline'], 'Mean'].droplevel(2)
difference_props['bottom']=means_props.loc[idx[:,:,'Baseline'], 'Mean'].droplevel(2)

####################################Alternative grapsh#########################
# ##################################Level first, then a seperate graph for differences
#        
####################################Proportions####################################
for var in propvariablelist:
    # do not droplevel 0 of index in case there is only 1 outcome. 
    if len(propvariablelist)==1:
        selmean=means_props.loc[idx[var,:,:]]#.droplevel(0)
    else:
        selmean=means_props.loc[idx[var,:,:]].droplevel(0)

    #draw out some parameters needed from nested dict
    param={}
    param=plotlabels_f_dict[var]
    #adding bottoms for means (this is equal to the minimum value from the param dict)
    selmean['bottom']=param['yminv']
    
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex='col',  figsize=(3, 3*1.61))
    axs=fig.axes
    #plot left hand graphs
    for i, cat in zip(range(0, 4, 1), catlist):
        
        #plot
        bars=axs[i].bar(x=selmean.loc[cat].index, height=selmean.loc[cat].Mean, width=0.5,  color=selmean.loc[cat].color, alpha=0.4, edgecolor=selmean.loc[cat].color, linewidth=2,
        yerr=selmean.loc[cat].err, ecolor=selmean.loc[cat].color)
        
        #labels
        autolabelpercent(bars, 'right')

        #horizontal helper line
        axs[i].axhline(y=selmean.loc[cat,'Baseline'].Mean, color='#44841A', ls=':')
        axs[i].axhline(y=selmean.loc[cat,'Endline'].Mean, color=selmean.loc[cat, 'Endline'].color, ls=':')
         
        #title
        axs[i].set_title(cat, color=cmaplabel[cat], fontsize=12, loc='left')


        #yaxis
        axs[i].set_ylim([0,1])
        axs[i].set_ylabel(None)
        axs[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


        #xaxis
        axs[i].get_xaxis().set_visible(True)
        axs[i].tick_params(axis='x', labelbottom=True, )



        #spines
        axs[i].spines['left'].set_visible(True)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines["left"].set_position(("outward", +5))

    fig.subplots_adjust(bottom=0.5, top=3)

    #footnotes
    plt.figtext(-0.3, 0.3, countryname + '\nVertical  bars represent 95% confidence intervals', size='small') 
    filename=alt_graphs_path/"{}.svg".format(var)
    plt.savefig(filename, dpi=300, facecolor='w', bbox_inches='tight')







######seperate horizontal plot for differences
for var in propvariablelist:
    # droplevel 1 of index (this is difference all the time)
    if len(propvariablelist)==1:
        seldif=difference_props.loc[idx[var,:,:]].droplevel(0).sort_values(by='category_nr')
    else:
        seldif=difference_props.loc[idx[var,:,:]].droplevel(0).droplevel(1).sort_values(by='category_nr')

    #non-significant bars are grey
    seldif['color']=np.where((seldif['pvalue']>0.05), '#d3d3d3', seldif['color'])


    fig, axes = plt.subplots(nrows=1, ncols=1,sharex='col',  figsize=(1.5, 1.5*1.61))
    axs=fig.axes         

    #plot
    i=0    # i=0 to keep code in case we need to iterate later
    bars=axs[i].barh(y=seldif.index, width=seldif.Mean, color=seldif.color, alpha=0.4, edgecolor=seldif.color, 
    xerr=seldif.err, ecolor=seldif.color)
        #title
    axs[i].set_title('Difference \n(endline-baseline)\nby activity', fontsize=12, loc='left')


    #labels (significant onle)
    #labels
    for bar in bars:
            width = bar.get_width()
            widthp=str(int(width*100))+ '%'
            axs[i].text(x= width*1.01,
            y= bar.get_y(),
            s= widthp,
            color=bar.get_edgecolor(),
            )
    
    #yaxis
    [t.set_color(c) for (t,c) in
    zip(axs[i].get_yticklabels(), list(seldif.color))]


    

    #xaxis
    axs[i].set_xlim([0,1])
    axs[i].set_xlabel(None)
    axs[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    #spines
    axs[i].spines['left'].set_position('zero')
    axs[i].spines['left'].set_visible(True)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)

    plt.figtext(0, -0.2, countryname + '\nVertical  bars represent 95% confidence intervals\nDifferences that are not statistically signficant at p<0.05 greyed out', size='small') 



    filename_a="{}.svg".format(var)
    filename_b='dif_'+filename_a
    filename=alt_graphs_path/filename_b
    plt.savefig(filename, dpi=300, facecolor='w', bbox_inches='tight')


        
####################################Scales####################################
for var in scalevariablelist:
    # do not droplevel 0 of index in case there is only 1 outcome. 
    if len(scalevariablelist)==1:
        selmean=means_scales.loc[idx[var,:,:]]#.droplevel(0)
    else:
        selmean=means_scales.loc[idx[var,:,:]].droplevel(0)
    #draw out some parameters needed from nested dict
    param={}
    param=plotlabels_f_dict[var]
    #adding bottoms for means (this is equal to the minimum value from the param dict)
    selmean['bottom']=param['yminv']
    
    fig, axes = plt.subplots(nrows=4, ncols=1,sharex='col',  figsize=(3, 3*1.61))
    axs=fig.axes
    #plot left hand graphs
    for i, cat in zip(range(0, 4, 1), catlist):
         #plot
        barsl=axs[i].bar(x=selmean.loc[cat].index, height=selmean.loc[cat].Mean, width=0.5,  color=selmean.loc[cat].color, alpha=0.4, edgecolor=selmean.loc[cat].color, 
        yerr=selmean.loc[cat].err, ecolor=selmean.loc[cat].color)
        
        #labels
        autolabel(barsl, 'right')
        
        #horizontal helper line
        axs[i].axhline(y=selmean.loc[cat,'Baseline'].Mean, color='#44841A', ls=':')
        axs[i].axhline(y=selmean.loc[cat,'Endline'].Mean, color=selmean.loc[cat, 'Endline'].color, ls=':')
 
        #title
        axs[i].set_title(cat, color=cmaplabel[cat], fontsize=12, loc='left')


        #yaxis
        axs[i].set_ylim(param['yminv'], param['ymaxv'])
        axs[i].set_ylabel(None)
        axs[i].set_yticks(np.arange(param['yminv'], param['ymaxv']+1, 1))
        ytick=axs[i].get_yticks().tolist()    
        ytick[0]=param['yminl']
        ytick[-1]=param['ymaxl']
        axs[i].set_yticklabels(ytick, fontsize=8)
        #xaxis
        axs[i].get_xaxis().set_visible(True)
        axs[i].tick_params(axis='x', labelbottom=True, )



        #spines
        axs[i].spines['left'].set_visible(True)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines["left"].set_position(("outward", +5))
        
    fig.subplots_adjust(bottom=0.5, top=3)

    #footnotes
    plt.figtext(-0.3, 0.3, countryname + '\nVertical  bars represent 95% confidence intervals', size='small') 

    filename=alt_graphs_path/"{}.svg".format(var)
    plt.savefig(filename, dpi=300, facecolor='w', bbox_inches='tight')




######seperate horizontal plot for differences SCALES
for var in scalevariablelist:
    # droplevel 1 of index (this is difference all the time)
    seldif=difference_scales.loc[idx[var,:,:]].droplevel(1).sort_values(by='category_nr')

    #non-significant bars are grey
    seldif['color']=np.where((seldif['pvalue']>0.05), '#d3d3d3', seldif['color'])


    fig, axes = plt.subplots(nrows=1, ncols=1,sharex='col',  figsize=(1.5, 1.5*1.61))
    axs=fig.axes         

    #plot
    #plot
    i=0    # i=0 to keep code in case we need to iterate later    # i=0 to keep code in case we need to iterate later
    bars=axs[i].barh(y=seldif.index, width=seldif.Mean, color=seldif.color, alpha=0.4, edgecolor=seldif.color, 
    xerr=seldif.err, ecolor=seldif.color)
            #title
    axs[i].set_title('Difference \n(endline-baseline)\nby activity', fontsize=12, loc='left')
    axs[i].tick_params(axis='y',labelleft=False, labelright=True)

    #labels (significant only)
    #labels
    for bar in bars:
            width = bar.get_width()
            widthp=round(width, 1)
            axs[i].text(x= width*1.01,
            y= bar.get_y(),
            s= widthp,
            color=bar.get_edgecolor(),
            )
    
    #yaxis
    [t.set_color(c) for (t,c) in
    zip(axs[i].get_yticklabels(), list(seldif.color))]


    

    #xaxis
    #axs[i].set_xlim([0,1])
    axs[i].set_xlabel(None)

    #spines
    axs[i].spines['left'].set_position('zero')
    axs[i].spines['left'].set_visible(True)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)

    plt.figtext(0, -0.2, countryname + '\nVertical  bars represent 95% confidence intervals\nDifferences that are not statistically signficant at p<0.05 greyed out', size='small') 

    filename_a="{}.svg".format(var)
    filename_b='dif_'+filename_a
    filename=alt_graphs_path/filename_b
    plt.savefig(filename, dpi=300, facecolor='w', bbox_inches='tight')


