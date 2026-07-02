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

    #find and read json files 
    json_dir  = os.path.join("./","merged_json")
    for i,mip in enumerate(mips):
      exp = exps[i] 
      ver = vers[i]
      json_list = sorted(glob.glob(os.path.join(json_dir,"*_{}_{}_{}.json".format(mip,exp,ver))))
      for json_file in json_list:
        print(json_file.split('/')[-1])
      if i == 0: 
        ref_case = Metrics(json_list)
        ref_name = "{}.{}".format(mip,exp)
      else: 
        tst_case = Metrics(json_list)
        tst_name = "{}.{}".format(mip,exp)
        version  = ver 
      del(json_list)

    print(ref_case.df_dict['rms_xy']['djf']['global']['model_run'])
    exit()

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
