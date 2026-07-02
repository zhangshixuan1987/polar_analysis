import glob
import os
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from pcmdi_metrics.graphics import Metrics
from pcmdi_metrics.graphics import normalize_by_median
from pcmdi_metrics.graphics import portrait_plot
from pcmdi_metrics.graphics import parallel_coordinate_plot
from pcmdi_metrics.graphics import TaylorDiagram

def taylor_plot(refname,refdic,tstname,testdic,statname,stat,varnames,season,region,
                ftype,diag_path,watermark):

  fig = plt.figure(figsize=(8,8))

  stddev = combined.df_dict["std_xy"][season][region][var].to_numpy()
  refstd = combined.df_dict['std-obs_xy'][season][region][var][0]
  corrcoeff = combined.df_dict["cor_xy"][season][region][var].to_numpy()
  models = combined.df_dict["cor_xy"][season][region]['model'].to_list()

  colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(models)))

  fig, ax = TaylorDiagram(stddev, corrcoeff, refstd, fig, colors, normalize=True, labels=models, ref_label='Ref: '+var_ref_dict[var])

  ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', ncol=2)
  fig.suptitle(', '.join([var, season, region]), fontsize=20)

  # Add Watermark
  if watermark:
    fig.text(0.5, 0.4, 'Example',
            fontsize=100, color='black', alpha=0.1,
            ha='center', va='center', rotation='25')

    # Save figure as an image file
    #save figure 
    figname = "{}_vs_{}_mean_climate_taylor_plot_{}_{}_{}.{}"
    figname = figname.format(tstname.upper(),refname.upper(),
                             region.upper(),season.upper(),
                             statname.upper(),ftype)
    if not os.path.exists(diag_path):
       os.makedirs(diag_path)
    fig.savefig(os.path.join(diag_path,figname), facecolor='w', bbox_inches='tight')
    return

def box_plot(refname,refdic,tstname,testdic,statname,stat,varnames,season,region,
             ftype,diag_path,watermark):
  
  print("Mean Climate Box Plots")
  fig = plt.figure(figsize=(14,14))
  ncols = 5
  nrows = int(math.ceil(len(var_list)/ncols))
  for index, var in enumerate(varnames):
    fig.add_subplot(nrows, ncols, index+1)
    # Box plot for library (i.e., CMIP6 models)
    ax = refdict[stat][season][region].boxplot(
        [var],
        # Customize box plot
        grid=False, fontsize=16,
        color=dict(boxes='black', whiskers='black', medians='black', caps='black'),
        boxprops=dict(linestyle='-', linewidth=1.5),
        flierprops=dict(linestyle='-', linewidth=1.5),
        medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
        whiskerprops=dict(linestyle='-', linewidth=1.5),
        capprops=dict(linestyle='-', linewidth=1.5),
        showfliers=False, # mute showing outliers
        rot=0,
        )
    # Add marker for test case (i.e. user models)
    try:
        my_model = tst_dict[stat][season][region][var]
        ax.plot(1, my_model, 'o', c='red', markersize=10, label='my_model')
    except:
        pass
    # Show unit as y-axis label
    ax.set_ylabel(var_unit_list[index].split('[')[-1].split(']')[0])
    # Show legend at upper right corner of the figure
    if index == ncols-1:
        h, l = ax.get_legend_handles_labels()
        black_patch = mpatches.Patch(color='black', label='CMIP6', fill=False)
        ax.legend(handles= [black_patch] + h,
                  bbox_to_anchor=(1, 1.02), loc='lower right', ncol=2)
    # Add Watermark
    if watermark: 
      fig.text(0.5, 0.4, 'Example',
               fontsize=100, color='black', alpha=0.1,
               ha='center', va='center', rotation='25')

    fig.suptitle('rms_xyt, ann, global', fontsize=20)
    fig.tight_layout(pad=1.5)

    #save figure 
    figname = "{}_vs_{}_mean_climate_box_plot_{}_{}_{}.{}"
    figname = figname.format(tstname.upper(),refname.upper(),
                             season.upper(),region.upper(),
                             statname.upper(),ftype)
    if not os.path.exists(diag_path):
      os.makedirs(diag_path)
    fig.savefig(os.path.join(diag_path,figname), facecolor='w', bbox_inches='tight')

def port4sea_plot(refname,tstname,statname,stat,data_dict,var_names,model_names,region,
                  ftype,diag_path,watermark,landscape,data_version):
    print("Portrait  Plots")
    if landscape:
      xaxis_labels = model_names
      yaxis_labels = var_names
      axis = 1
      figsize = (40,18) #(40, 20)
    else:
      xaxis_labels = var_names
      yaxis_labels = model_names
      axis = 0
      figsize = (18, 25)    
    
    # extract data for 4 seasons
    data_all = dict()
    for season in ['djf','mam','jja','son']:
      if landscape:
        data_all[season] = data_dict[stat][season][region][var_names].to_numpy().T
      else:
        data_all[season] = data_dict[stat][season][region][var_names].to_numpy()
    if stat != "cor_xy":
      # normalize data by median value 
      data_djf_nor = normalize_by_median(data_all['djf'], axis=axis)
      data_mam_nor = normalize_by_median(data_all['mam'], axis=axis)
      data_jja_nor = normalize_by_median(data_all['jja'], axis=axis)
      data_son_nor = normalize_by_median(data_all['son'], axis=axis)
    else:
      # correlation 
      data_djf_nor = data_all['djf']
      data_mam_nor = data_all['mam']
      data_jja_nor = data_all['jja']
      data_son_nor = data_all['son']

    data_all_nor = np.stack([data_djf_nor, data_mam_nor, data_jja_nor, data_son_nor])
    data_all_nor.shape

    if stat != "cor_xy":
      cbar_label   = "{}, normalized by median".format(statname.upper())
      var_range    = (-0.5,0.5)
      cmap_bounds  = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    else:
      cbar_label   = "Pattern Corr."
      var_range    = (-1,1)
      cmap_bounds  = [0.1, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.0]

    print('len(var_list): ', len(var_names))
    print('len(model_names): ', len(model_names))    
    print('data_nor.shape:', data_all_nor.shape)
    
    if landscape:
      legend_box_xy=(1.1, 1.2)
      legend_box_size=4
      legend_lw=1
      legend_fontsize=12.5 * 1.2
    else:
      legend_box_xy=(1.25, 1)
      legend_box_size=3
      legend_lw=1
      legend_fontsize=12.5 * 1.5

    logo_rect = [0.85, 0.15, 0.07, 0.07]
    logo_off  = True

    #Using Matplotlib-based PMP Visualization Function to Generate Portrait Plot
    fig, ax, cbar = portrait_plot(data_all_nor,
                                  xaxis_labels=xaxis_labels,
                                  yaxis_labels=yaxis_labels,
                                  cbar_label=cbar_label,
                                  box_as_square=True,
                                  vrange=var_range,
                                  figsize=figsize,
                                  cmap='RdYlBu_r',
                                  cmap_bounds=cmap_bounds,
                                  cbar_kw={"extend": "both"},
                                  missing_color='grey',
                                  legend_on=True,
                                  legend_labels=['DJF', 'MAM', 'JJA', 'SON'],
                                  legend_box_xy=legend_box_xy, 
                                  legend_box_size=legend_box_size,
                                  legend_lw=legend_lw,
                                  legend_fontsize=legend_fontsize,
                                  #logo_rect = logo_rect
                                  logo_off  = logo_off
                                 )

    ax.set_xticklabels(xaxis_labels, rotation=45, va='bottom', ha="left")
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    cbar.ax.tick_params(labelsize=18) 
    #cbar.ax.set_title(cbar_label,fontsize=20)
    #plt.rcParams['legend.title_fontsize'] = 20

    # Add title
    ax.set_title("Seasonal climatology Model Performance ({})".format(region.upper()), fontsize=30, pad=30)

    # Add data info
    #fig.text(1.25, 0.9, 'Data version\n'+data_version, transform=ax.transAxes,
    #         fontsize=12, color='black', alpha=0.6, ha='left', va='top',)

    # Add Watermark
    if watermark:
      ax.text(0.5, 0.5, 'E3SM-PCMDI Example', transform=ax.transAxes,
              fontsize=100, color='black', alpha=0.5,
              ha='center', va='center', rotation=25)

    # Save figure as an image file
    #save figure 
    figname = "{}_vs_{}_mean_climate_portrait_plot_4seasons_{}_{}.{}"
    figname = figname.format(tstname.upper(),refname.upper(),region.upper(),
                             statname.upper(),ftype)
    if not os.path.exists(diag_path):
       os.makedirs(diag_path)
    fig.savefig(os.path.join(diag_path,figname), facecolor='w', bbox_inches='tight')
    return 

def paracord_plot(refname,tstname,statname,stat,ref_dic,tst_dic,df_dic,hlmodels,season,varnames,region,
                  ftype,diag_path,watermark):
    print("Parallel Coordinate Plots")
    units_all = 'prw [kg m-2], pr [mm d-1], psl [Pa], rlds [W m-2], rsdscs [W m-2], rltcre [W m-2], rlus [W m-2], rlut [W m-2], rlutcs [W m-2], rsds [W m-2], rsdt [W m-2], rstcre [W m-2], rsus [W m-2], rsut [W m-2], rsutcs [W m-2], sfcWind [m s-1], zg-500 [m], ta-200 [K], ta-850 [K], tas [K], ts [K], ua-200 [m s-1], ua-850 [m s-1], uas [m s-1], va-200 [m s-1], va-850 [m s-1], vas [m s-1], tauu [Pa], tauv [Pa], hfls [W m-2], hfss [W m-2], prsn [mm d-1]'
    units_all.split(', ')
    var_unit_list = []
    for var in varnames:
      found = False
      for var_units in units_all.split(', '):
        tmp1 = var_units.split(' [')[0]
        #print(var, tmp1)
        if tmp1 == var:
          unit = '[' + var_units.split(' [')[1]
          var_unit_list.append(var + '\n' + unit)
          found = True
          break
      if found is False:
        print(var, 'not found')
        varnames.remove(var)
    print('var_unit_list:', var_unit_list)
    metric_names = var_unit_list
    #print(df_dic[stat][season][region])
    data = df_dic[stat][season][region][varnames].to_numpy()
    ref_names   = ref_dic[stat][season][region]['model_run'].tolist()
    tst_names   = tst_dic[stat][season][region]['model_run'].tolist()
    ref_names   = [ ref + "(amip)" for ref in ref_names]
    tst_names   = [ tst + "(historical)" for tst in tst_names]
    model_names = ref_names + tst_names + ['E3SM AMIP mean', "E3SM historical mean"]
    #model_names = df_dic[stat][season][region]['model_run'].tolist()

    print(data)
    #metric_names = ['\n['.join(var_unit.split(' [')) for var_unit in var_unit_list]
    #metric_names = var_list
    model_highlights = list(hlmodels) 
    print(model_highlights)
    print('data.shape:', data.shape)
    print('len(metric_names): ', len(metric_names))
    print('len(model_names): ', len(model_names))

    title = 'Mean Climate: {}, {}, {}'.format(statname.upper(),season.upper(),region.upper())
    fig, ax = parallel_coordinate_plot(data, metric_names, model_names, model_highlights,
                                       title=title,
                                       figsize=(21, 7),
                                       colormap='tab20',
                                       show_boxplot=False,
                                       show_violin=True,
                                       xtick_labelsize=10,
                                       logo_rect=[0.8, 0.8, 0.15, 0.15])
    #fig.text(0.99, -0.45, 'Data version\n'+data_version, transform=ax.transAxes,
    #         fontsize=12, color='black', alpha=0.6, ha='right', va='bottom',)

    if (watermark):
      # Add Watermark
      ax.text(0.5, 0.5, 'Example', transform=ax.transAxes,
              fontsize=100, color='black', alpha=0.2,
              ha='center', va='center', rotation=25)

    # Save figure as an image file
    #save figure 
    figname = "{}_vs_{}_mean_climate_Parallel_Coordinate_plot_{}_{}_{}.{}"
    figname = figname.format(tstname.upper(),
                             refname.upper(),
                             season.upper(),
                             region.upper(),
                             statname.upper(),
                             ftype)
    if not os.path.exists(diag_path):
       os.makedirs(diag_path)
    fig.savefig(os.path.join(diag_path,figname), facecolor='w', bbox_inches='tight')

    del(data,model_names,model_highlights,metric_names,units_all,var_unit_list)
    return

# Defining main function
def main():
    # path to find code and output 
    pcmdi_dir       = "/lcrc/group/acme/ac.szhang/acme_scratch/pcmdi_metrics"
    pcmdi_src_path  = os.path.join(pcmdi_dir,"pcmdi_metrics")
    pcmdi_par_path  = os.path.join(pcmdi_dir,"e3sm_setup")
    pcmdi_data_path = "/lcrc/group/acme/ac.szhang/acme_scratch/data/pcmdi"

    ftype = "pdf" # figure type, pdf, png ...
    #diagnostic output path 
    pcmdi_diag_out  = "./" #os.path.join(".","mean_climate") 

    #groups to be compared 
    mips    = ["cmip6",       "e3sm"      ]    
    exps    = ["historical",  "historical"]
    vers    = ["v20230823",   "v20240810" ]

    mips    = ["e3sm",        "e3sm"      ]
    exps    = ["amip",        "historical"]
    vers    = ["v20240810",   "v20240810" ]

    #mips    = ["analysis",    "e3sm"      ]
    #exps    = ["historical",  "historical"]
    #vers    = ["v20240810",   "v20240810" ]

    #custom variable file 
    var_dic = json.load(open(os.path.join(".",'custom_mean_clim_metric.json')))["mean_climate"]
    viv_dic = {v: k for k, v in var_dic["vars"].items()}
    vars = list(var_dic["vars"].keys())
    #print(vars)
    #exit()
    vmod = []
    unts = []
    for var in vars:
      vmod.append(var_dic["vars"].get(var,var))
      unts.append(var_dic["units"].get(var,var))
    #vars = ['pr', 'prw', 'psl', 'rlds', 'rltcre', 'rlut', 'rlutcs', 'rsds', 'rsdt', 'rstcre', 'tas', 'ts']
    #vmod = vars
    #unts = []
    #for var in vars:
    #  unts.append(var_dic["units"].get(var,var))
    #del(var_dic) 

    #find and read json files 
    json_dir  = os.path.join("./","merged_json")
    for i,mip in enumerate(mips):
      exp = exps[i] 
      ver = vers[i]
      json_list = sorted(glob.glob(os.path.join(json_dir,"*_{}_{}_{}.json".format(mip,exp,ver))))
      for json_file in json_list:
        print(json_file.split('/')[-1])
      #print(json_dir)
      #print(json_list)
      if i == 0: 
        ref_case = Metrics(json_list)
        ref_name = "{}.{}".format(mip,exp)
      else: 
        tst_case = Metrics(json_list)
        tst_name = "{}.{}".format(mip,exp)
        version  = ver 
      del(json_list)

    #prepar data for plots
    combined = ref_case.merge(tst_case)
    df_dict = combined.df_dict
    var_list = sorted(combined.var_list)
    var_unit_list = combined.var_unit_list
    var_ref_dict = combined.var_ref_dict
    regions = combined.regions
    stats = combined.stats
    print('var_list:', sorted(var_list))
    print('var_unit_list:', sorted(var_unit_list))
    print('var_ref_dict:', var_ref_dict)
    print('regions:', regions)
    print('stats:', stats)
    print('seasons:', df_dict["rms_xy"].keys())

    ############################################
    #plot Portrait Plot with 4 Triangles (4 seasons)
    var_list  = list(vars)
    diag_path = os.path.join(pcmdi_diag_out,"Portrait_4season")
    landscape = True 
    watermark = False
    for region in ["global"]: #,"SH","NH"]:
      ref_names   = ref_case.df_dict['rms_xyt']['ann'][region]['model_run'].tolist()
      tst_names   = tst_case.df_dict['rms_xyt']['ann'][region]['model_run'].tolist()
      ref_names   = [ ref + "(amip)" for ref in ref_names]
      tst_names   = [ tst + "(historical)" for tst in tst_names]
      model_names = ref_names + tst_names
      #model_names = df_dict['rms_xyt']['ann'][region]['model_run'].tolist()
      
      port4sea_plot(ref_name,tst_name,"RMSE","rms_xy",df_dict,var_list,
                    model_names,region,ftype,diag_path,watermark,landscape,version)
      port4sea_plot(ref_name,tst_name,"CORR","cor_xy",df_dict,var_list,
                    model_names,region,ftype,diag_path,watermark,landscape,version)
    del(diag_path)

    ############################################
    #plot Parallel Coordinate Plot
    var_list = ['pr', 'prw','psl','prsn','hfss','hfls','rlds','rltcre','rstcre',
                'ta-850','ua-200','ua-850','va-200','va-850','zg-500']
    print('var_list:', sorted(var_list))
    print('var_unit_list:', sorted(var_unit_list))
    print('var_ref_dict:', var_ref_dict)
    diag_path = os.path.join(pcmdi_diag_out,"Parallel_Coordinate")
    for region in ["global"]: #,"SH","NH"]:
      for season in ["djf","jja","mam","son"]:
        #hlmodels = tst_case.df_dict["rms_xy"][season][region]['model_run'].to_list()
        #hlmodels = ['v3-LR-0151']
        # mean value of statistics from multi models in each CMIP
        df_dict["rms_xy"][season][region].loc['AMIP mean'] = ref_case.df_dict["rms_xy"][season][region].mean(numeric_only=True, skipna=True)
        df_dict["rms_xy"][season][region].loc['HIST mean'] = tst_case.df_dict["rms_xy"][season][region].mean(numeric_only=True, skipna=True)
        df_dict["rms_xy"][season][region].at['AMIP mean', 'model'] = 'E3SM AMIP mean'
        df_dict["rms_xy"][season][region].at['HIST mean', 'model'] = 'E3SM historical mean'
        hlmodels = ['E3SM AMIP mean', 'E3SM historical mean']
        paracord_plot(ref_name,tst_name,"RMSE","rms_xy",ref_case.df_dict,tst_case.df_dict,df_dict,hlmodels,season,var_list,region,ftype,diag_path,watermark)
    del(df_dict,var_list)

    ############################################
    #plot box plot 
    var_list = ['hfls', 'hfss', 'pr', 'prsn', 'psl', 'rltcre', 'rlut', 
                'rstcre', 'sfcWind', 'ta-850', 'tas', 'tauu', 'tauv', 'ts']
    print('var_list:', sorted(var_list))
    print('var_unit_list:', sorted(var_unit_list))
    print('var_ref_dict:', var_ref_dict)
    diag_path = os.path.join(pcmdi_diag_out,"Box_Plots")
    for season in ["ann"]: #,"djf","jja","mam","son"]:
      box_plot(ref_name,ref_case,tst_name,tst_case,"RMSE","rms_xyt",var_list,season,region,
               ftype,diag_path,watermark)

# Using the special variable
if __name__== "__main__":
    main()
