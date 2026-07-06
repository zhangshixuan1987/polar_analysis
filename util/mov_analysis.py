import os
import glob
import json
import numpy as np
import pandas as pd
from pcmdi_metrics.graphics import parallel_coordinate_plot, portrait_plot

# ============================================================
# Load metric files from variability modes
# ============================================================
def load_metric_file_list_mov(path, mip, exp, group, case_id, modes, bases, eofs): 
    file_list = []
    for i, mode in enumerate(modes):
        base = bases[i]
        eof = eofs[i]
        ftest = glob.glob(os.path.join(path, mip, exp, case_id, mode, base,
            "var_mode_{}_{}_*_{}_{}_*_{}_*_cmec.json".format(mode, eof, mip, exp, group)))
        for ff in ftest:
            if os.path.exists(ff):
                file_list.append(ff)
    return file_list

def load_metrics_data_mov(mip, json_list, key, eofs, modes):
    data_lib = {}
    key1 = 'RESULTS'
    key2 = 'defaultReference'
    for jsonObj in json_list:
        with open(jsonObj) as json_file:
            dataDict = json.load(json_file)
            if mip == "e3sm":
                groups = [key]
            else: 
                groups = dataDict[key1].keys()
            for group in groups:
                if mip == "e3sm":
                    exps = dataDict[key1][group].keys() 
                else:
                    exps = [key]
                for exp in exps:
                    for mode in dataDict[key1][group][exp][key2].keys():
                        eof = eofs[modes.index(mode)].lower()
                        for season in dataDict[key1][group][exp][key2][mode].keys():
                            if season != "attributes":
                                for metric in dataDict[key1][group][exp][key2][mode][season][eof].keys():
                                    if metric != "cbf":
                                        if metric not in data_lib.keys():
                                            data_lib[metric] = {}
                                        if group not in data_lib[metric].keys():
                                            data_lib[metric][group] = {}
                                        if exp not in data_lib[metric][group].keys():
                                            data_lib[metric][group][exp] = {}
                                        if mode not in data_lib[metric][group][exp].keys():
                                            data_lib[metric][group][exp][mode] = {}
                                        if season not in data_lib[metric][group][exp][mode].keys():
                                            data_lib[metric][group][exp][mode][season] = {}
                                        data_lib[metric][group][exp][mode][season] = \
                                            dataDict[key1][group][exp][key2][mode][season][eof][metric]
     
    out_lib = {}
    highlight_models = []
    for metric in data_lib.keys():
        if metric not in out_lib.keys():
            out_lib[metric] = {}
        for group in data_lib[metric].keys():
            for exp in data_lib[metric][group].keys():
                dtmp = [group, exp, '{}_{}'.format(group, exp)]
                title = ['model', 'run', 'model_run']
                for mode in data_lib[metric][group][exp].keys(): 
                    for season in data_lib[metric][group][exp][mode].keys():
                        dtmp.append(data_lib[metric][group][exp][mode][season])
                        k = '{}({})'.format(mode, season)
                        if k not in title: 
                            title.append(k)
                if 'df' not in locals():
                    df = pd.DataFrame([dtmp], columns=title)
                else:
                    df = pd.concat([df, pd.DataFrame([dtmp], columns=title)], ignore_index=True)
                del (dtmp) 

        if mip == "cmip":
            for model in df['model']:
                if ("e3sm" in model.lower()) and (model not in highlight_models):
                    highlight_models.append(model)
            for model in highlight_models:
                idxs = df[df.iloc[:, 0] == model].index
                for idx in idxs:
                    igx = [i for i in df.index if i != idx]
                    df = df.loc[igx + [idx]]
            
        out_lib[metric] = df
        if 'df' in locals():
            del (df)
    return out_lib, highlight_models

# ============================================================
# Load metric files from ENSO
# ============================================================
def load_metric_file_list_enso(path, mip, exp, key, case_id, modes, bases, eofs): 
    mode = modes[modes.index('ENSO')]
    base = bases[modes.index('ENSO')]
    metric_collection = eofs[modes.index('ENSO')]
    file_list = []
    if 'cmip' in mip: 
        ftest = glob.glob(os.path.join(path, mip, exp, case_id, mode, base,
            "{}_{}_{}_*_{}.json".format(mip, exp, metric_collection, key)))  
    else:
        ftest = glob.glob(os.path.join(path, mip, exp, case_id, mode, base,
            "{}_{}_{}_*_{}_*.json".format(mip, exp, metric_collection, key)))
    for ff in ftest:
        if os.path.exists(ff) and "diveDown" not in ff:
            file_list.append(ff)
    return file_list

def load_metrics_data_enso(mip, json_list, key, eofs, bases, modes):
    mode = modes[modes.index('ENSO')]
    metric_collection = eofs[modes.index('ENSO')]

    data_lib = {}
    for jsonObj in json_list:
        with open(jsonObj) as json_file:
            dataDict = json.load(json_file)['RESULTS']['model']
            if mip == "e3sm":
                groups = [key]
            else:
                groups = dataDict.keys()
            for group in groups:
                if mip == "e3sm":
                    exps = dataDict[group].keys()
                else:
                    exps = [key]
                for exp in exps:
                    for metric in dataDict[group][exp]["value"].keys(): 
                        if metric_collection == "ENSO_tel" and "Map" in metric:
                            dict_dia = dataDict[group][exp]["value"][metric+"Corr"]["diagnostic"]
                            diagnostic_values = dict((key1, None) for key1 in dict_dia.keys())
                            diagnostic_units = ""
                        else:
                            dict_dia = dataDict[group][exp]["value"][metric]["diagnostic"]
                            diagnostic_values = dict((key1, dict_dia[key1]["value"]) for key1 in dict_dia.keys())
                            diagnostic_units = dataDict[group][exp]["metadata"]["metrics"][metric]["diagnostic"]["units"]
                            
                        if metric_collection == "ENSO_tel" and "Map" in metric:
                            list1, list2 = [metric+"Corr", metric+"Rmse"], ["diagnostic", "metric"]
                            dict_met = dataDict[group][exp]["value"]
                            metric_values = dict((key1, {model: [dict_met[su][ty][key1]["value"] for su, ty in zip(list1, list2)]})
                                             for key1 in dict_met[list1[0]]["metric"].keys())
                            metric_units = [dataDict[group][exp]["metadata"]["metrics"][su]["metric"]["units"] for su in list1]
                        else:
                            dict_met = dataDict[group][exp]["value"][metric]["metric"]
                            metric_values = dict((key1, {exp: dict_met[key1]["value"]}) for key1 in dict_met.keys())
                            metric_units = dataDict[group][exp]["metadata"]["metrics"][metric]["metric"]["units"]
                            
                        if metric not in data_lib.keys():
                            data_lib[metric] = {}
                        if group not in data_lib[metric].keys():
                            data_lib[metric][group] = {}
                        if exp not in data_lib[metric][group].keys():
                            data_lib[metric][group][exp] = {}
                        for ff in diagnostic_values.keys():
                            gg = "{}_{}".format(group, exp)
                            if ff == gg:
                                data_lib[metric][group][exp]['diagnostic_model'] = diagnostic_values[gg]
                                data_lib[metric][group][exp]['diagnostic_unit'] = diagnostic_units
                            else:
                                data_lib[metric][group][exp]['diagnostic_obs'] = diagnostic_values[ff]  
                                data_lib[metric][group][exp]['metric_value'] = metric_values[ff][exp] 
                                data_lib[metric][group][exp]['metric_unit'] = metric_units
                                
    out_lib = {}
    highlight_models = []
    for metric in data_lib.keys():
        if metric not in out_lib.keys():
            out_lib[metric] = {}
        for group in data_lib[metric].keys():
            for exp in data_lib[metric][group].keys():
                dtmp = [group, exp, '{}_{}'.format(group, exp),
                        data_lib[metric][group][exp].get('metric_value', np.nan)]
                title = ['model', 'run', 'model_run', metric]
                if 'df' not in locals():
                    df = pd.DataFrame([dtmp], columns=title)
                else:
                    df = pd.concat([df, pd.DataFrame([dtmp], columns=title)], ignore_index=True)
                del (dtmp)
        if mip == "cmip":
            for model in df['model']:
                if ("e3sm" in model.lower()) and (model not in highlight_models):
                    highlight_models.append(model)
            for model in highlight_models:
                idxs = df[df.iloc[:, 0] == model].index
                for idx in idxs:
                    igx = [i for i in df.index if i != idx]
                    df = df.loc[igx + [idx]]
        out_lib[metric] = df
        if 'df' in locals():
            del (df)
    return out_lib, highlight_models

# ============================================================
# Main loaders and consolidators
# ============================================================
def load_metric_file_list_asl(path, bases, modes, indices, period, seasons):
    file_list = {}
    for i, mode in enumerate(modes):
        refobs = bases[i]
        for sea in seasons: 
            if sea in ['monthly']: 
                ftest = glob.glob(os.path.join(path, refobs, "{}*{}.{}.csv".format(mode, 'AC', period)))
            else: 
                ftest = glob.glob(os.path.join(path, refobs, "{}*{}.{}.csv".format(mode, sea, period)))
            for ff in ftest:
                if os.path.exists(ff):
                    fname = ff.split("/")[-1]
                    metric = fname.split(".")[1]
                    if mode not in file_list.keys():
                        file_list[mode] = {}
                    if metric not in file_list[mode].keys():
                        file_list[mode][metric] = {}
                    file_list[mode][metric][sea] = ff 
    return file_list

def load_metrics_data_asl(csv_list, modes, indices, seasons):
    highlight_models = []
    test_models = []
    out_lib = {}
    for mode in csv_list.keys():
        out_lib[mode] = {}
        for metric in csv_list[mode].keys():
            out_lib[mode][metric] = {}
            for sea in seasons:
                csvObj = csv_list[mode][metric][sea]
                df = pd.read_csv(csvObj)
                if 'df_all' not in locals():
                    df_all = df.copy()
                    df_all.insert(1, "season", sea)
                else:
                    df_tmp = df.copy()
                    df_tmp.insert(1, "season", sea)
                    df_all = pd.concat([df_all, df_tmp], ignore_index=True)
                    del (df_tmp)
            
            for sea in seasons:
                df_sea = df_all[df_all['season'] == sea].copy()
                df_sea = df_sea.drop(columns=['season'])
                out_lib[mode][metric][sea] = df_sea
                del (df_sea)
            del (df_all)
    return out_lib, highlight_models, test_models

# ============================================================
# Saving CSV utility
# ============================================================
def save_figure_data(stat, var_names, var_units, data_dict, template, outdir):
    fname = template.replace("%(metric)", stat)
    outfile = os.path.join(outdir, fname)
    outdic = pd.DataFrame(data_dict)
    if "model_run" in outdic.columns:
        outdic = outdic.drop(columns=["model_run"])
    for var in list(outdic.columns.values[3:]):
        if var not in var_names:
            outdic = outdic.drop(columns=[var])
        else:
            outval = var_units[var_names.index(var)]
            outdic.columns.values[outdic.columns.values.tolist().index(var)] = outval
    outdic.to_csv(outfile, index=False)

def run_gen_data_enso(pmp_path, bases, modes, eofs, period, output_path):
    print("Consolidating ENSO metrics...")
    cmip_json = load_metric_file_list_enso(pmp_path, "cmip6", "historical", "r1i1p1f1", "v20241111", modes, bases, eofs)
    e3sm_json = load_metric_file_list_enso(pmp_path, "e3sm", "historical", "v2_1-SORRM", "v20241111", modes, bases, eofs)
    
    cmip_dict, cmip_hl = load_metrics_data_enso("cmip", cmip_json, "historical", eofs, bases, modes)
    e3sm_dict, _ = load_metrics_data_enso("e3sm", e3sm_json, "v2_1-SORRM", eofs, bases, modes)
    
    file_template = "ENSO.metrics.%(metric).{}.csv".format(period)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for metric in cmip_dict.keys():
        df_cmip = cmip_dict[metric]
        df_e3sm = e3sm_dict.get(metric, pd.DataFrame())
        df_merged = pd.concat([df_cmip, df_e3sm], ignore_index=True)
        
        # Calculate MME and MMM
        df_sub1 = df_merged[df_merged['run'] == "r1i1p1f1"]
        df_sub2 = df_merged[df_merged['run'] != "r1i1p1f1"]
        df_merged.loc['CMIP6_MME'] = df_sub1.mean(numeric_only=True, skipna=True)
        df_merged.at['CMIP6_MME', 'model'] = 'CMIP6'
        df_merged.at['CMIP6_MME', 'run'] = 'MMM'
        df_merged.at['CMIP6_MME', 'model_run'] = 'CMIP_MMM'
        
        df_merged.loc['v21_SORRM'] = df_sub2.mean(numeric_only=True, skipna=True)
        df_merged.at['v21_SORRM', 'model'] = 'v2_1-SORRM'
        df_merged.at['v21_SORRM', 'run'] = 'MMM'
        df_merged.at['v21_SORRM', 'model_run'] = 'v2_1-SORRM_MMM'
        
        df_merged = df_merged.fillna(value=np.nan)
        save_figure_data(metric, [metric], [metric], df_merged, file_template, output_path)

def run_gen_data_mov(pmp_path, bases, modes, eofs, period, output_path):
    print("Consolidating Variability Modes metrics...")
    cmip_json = load_metric_file_list_mov(pmp_path, "cmip6", "historical", "r1i1p1f1", "v20241111", modes, bases, eofs)
    e3sm_json = load_metric_file_list_mov(pmp_path, "e3sm", "historical", "v2_1-SORRM", "v20241111", modes, bases, eofs)
    
    cmip_dict, cmip_hl = load_metrics_data_mov("cmip", cmip_json, "historical", eofs, modes)
    e3sm_dict, _ = load_metrics_data_mov("e3sm", e3sm_json, "v2_1-SORRM", eofs, modes)
    
    file_template = "MOV.metrics.%(metric).{}.csv".format(period)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for metric in cmip_dict.keys():
        df_cmip = cmip_dict[metric]
        df_e3sm = e3sm_dict.get(metric, pd.DataFrame())
        df_merged = pd.concat([df_cmip, df_e3sm], ignore_index=True)
        
        df_sub1 = df_merged[df_merged['run'] == "r1i1p1f1"]
        df_sub2 = df_merged[df_merged['run'] != "r1i1p1f1"]
        df_merged.loc['CMIP6_MME'] = df_sub1.mean(numeric_only=True, skipna=True)
        df_merged.at['CMIP6_MME', 'model'] = 'CMIP6'
        df_merged.at['CMIP6_MME', 'run'] = 'MMM'
        df_merged.at['CMIP6_MME', 'model_run'] = 'CMIP_MMM'
        
        df_merged.loc['v21_SORRM'] = df_sub2.mean(numeric_only=True, skipna=True)
        df_merged.at['v21_SORRM', 'model'] = 'v2_1-SORRM'
        df_merged.at['v21_SORRM', 'run'] = 'MMM'
        df_merged.at['v21_SORRM', 'model_run'] = 'v2_1-SORRM_MMM'
        
        df_merged = df_merged.fillna(value=np.nan)
        
        var_list = list(df_cmip.columns[3:])
        save_figure_data(metric, var_list, var_list, df_merged, file_template, output_path)

def run_gen_data_asl(pmp_path, bases, modes, indices, period, seasons, output_path):
    print("Consolidating ASL metrics...")
    file_list = load_metric_file_list_asl(pmp_path, bases, modes, indices, period, seasons)
    merge_dict, highlight_models, test_models = load_metrics_data_asl(file_list, modes, indices, seasons)
    
    file_template = "ASL.metrics.%(metric).{}.csv".format(period)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for mode in merge_dict.keys():
        for metric in merge_dict[mode].keys():
            df_merged = merge_dict[mode][metric]
            
            df_sub1 = df_merged[df_merged['run'] == "r1i1p1f1"]
            df_sub2 = df_merged[df_merged['run'] != "r1i1p1f1"]
            df_merged.loc['CMIP6_MME'] = df_sub1.mean(numeric_only=True, skipna=True)
            df_merged.at['CMIP6_MME', 'model'] = 'CMIP6'
            df_merged.at['CMIP6_MME', 'run'] = 'MMM'
            df_merged.at['CMIP6_MME', 'model_run'] = 'CMIP_MMM'
            
            df_merged.loc['v21_SORRM'] = df_sub2.mean(numeric_only=True, skipna=True)
            df_merged.at['v21_SORRM', 'model'] = 'v2_1-SORRM'
            df_merged.at['v21_SORRM', 'run'] = 'MMM'
            df_merged.at['v21_SORRM', 'model_run'] = 'v2_1-SORRM_MMM'
            
            df_merged = df_merged.fillna(value=np.nan)
            var_list = list(df_merged.columns[3:])
            save_figure_data(metric, var_list, var_list, df_merged, file_template, output_path)
