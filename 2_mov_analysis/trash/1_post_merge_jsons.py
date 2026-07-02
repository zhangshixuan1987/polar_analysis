#!/usr/bin/env python
import copy
import glob
import json
import os

from pcmdi_metrics.utils import StringConstructor
from pcmdi_metrics.variability_mode.lib import dict_merge

def main():

  pmprdir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi"
  outdir  = "./merged_json" 
  case_id = "v20240824"
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  

  # variables to be used in mean climate metrics 
  var_dic = json.load(open(os.path.join("./",'custom_mean_clim_metric.json')))["mean_climate"]
  viv_dic = {v: k for k, v in var_dic["vars"].items()}

  mips    = ["cmip6",       "analysis",   "e3sm",      "e3sm"      ]
  exps    = ["historical",  "historical", "amip",      "historical"]
  cids    = ["v20240810",   "v20240810",  "v20240810", "v20240810" ]
  syear   = 1985
  eyear   = 2014

  obs_selection = "default"

  #find and merge jason 
  for var in var_dic["vars"]:
    # Load individual JSON and merge to one big dictionary
    vmod = var_dic["vars"].get(var,var)
    print(var,vmod)
    for k,mip in enumerate(mips):
      exp = exps[k]
      ver = cids[k] 
      json_dir  = os.path.join(pmprdir,"metrics_results","mean_climate",mip,exp,cids[k])
      json_files = sorted(glob.glob(os.path.join(json_dir,vmod+"_*_"+cids[k]+'.json'))) 
      print(json_dir)
      for j, json_file in enumerate(json_files):
        print(j, json_file)
        f = open(json_file)
        dict_tmp = json.loads(f.read())
        #rename variable for consistency 
        if mip not in ["cmip6_pcmdi", "cmip5_pcmdi"]:
          if len(var.split("-")) > 1: 
            dict_tmp['Variable']['id'] = var.split("-")[0] 
            dict_tmp['Variable']['level'] = float(var.split("-")[1])*100.0
        if j == 0:
          dict_final = dict_tmp.copy()
        else:
          dict_merge(dict_final, dict_tmp)
        f.close()
        del(dict_tmp)

      # Dump final dictionary to JSON
      final_json_filename = StringConstructor("%(var)_%(mip)_%(exp)_%(case_id).json")(
         var=var, mip=mip.split("_")[0], exp=exp, case_id=ver
      )
      final_json_file = os.path.join(outdir,final_json_filename)

      with open(final_json_file, "w") as fp:
        json.dump(dict_final, fp, sort_keys=True, indent=4)
      del(json_dir,json_files)

if __name__ == "__main__":
  main()
