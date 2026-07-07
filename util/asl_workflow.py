import os
import glob
import copy


MODEL_SUITES = {
    "v2_1_LR": {
        "description": "E3SM v2.1 LR historical ensemble",
        "case_keys": [
            "v2.1-LR-HIST-0101",
            "v2.1-LR-HIST-0151",
            "v2.1-LR-HIST-0201",
            "v2.1-LR-HIST-0251",
        ],
        "member_alias_by_case": {
            "v2.1-LR-HIST-0101": "en01",
            "v2.1-LR-HIST-0151": "en02",
            "v2.1-LR-HIST-0201": "en03",
            "v2.1-LR-HIST-0251": "en04",
        },
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["historical"],
        "input_source": "cvdp_rgd",
        "use_pcmdi_model_input": True,
        "control_case_key": "v2.1-LR-PiControl",
        "control_experiment": "piControl",
        "control_member": "en00",
        "control_period": "lr_piControl",
        "cvdp_rgd_root": "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/CVDP_RGD",
        "cvdp_rgd_experiment_dirs": {
            "historical": "v2_1.LR.historical",
            "piControl": "v2_1.LR.piControl",
        },
        "clim_mips": ["analysis", "e3sm"],
        "clim_exps": ["historical", "piControl"],
        "clim_test_mips": ["e3sm"],
        "clim_test_exps": ["historical"],
    },
    "v2_1_SORRM": {
        "description": "E3SM v2.1 SORRM historical+SSP370 ensemble",
        "selection": {
            "model_version": "v2.1",
            "configuration": "SORRM",
            "experiment": "historical+ssp370",
            "variant": "standard",
        },
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["historical", "ssp370"],
        "use_pcmdi_model_input": False,
        "control_case_key": "v2.1-SORRM-Control",
        "control_experiment": "piControl",
        "control_member": "0001",
        "control_period": "sorrm_piControl",
        "clim_mips": ["analysis", "e3sm"],
        "clim_exps": ["historical", "ssp370", "piControl"],
        "clim_test_mips": ["e3sm"],
        "clim_test_exps": ["historical"],
    },
    "v3_LR": {
        "description": "E3SM v3 LR historical+SSP245 ensemble",
        "case_keys": [f"v3-LR-HISTSSP245-{member:02d}" for member in range(1, 25)],
        "member_alias_by_case": {
            f"v3-LR-HISTSSP245-{member:02d}": f"en{member:02d}"
            for member in range(1, 25)
        },
        "case_metadata_by_case": {
            f"v3-LR-HISTSSP245-{member:02d}": {
                "model_name": "v3.LR.historical-SSP245",
                "case_name": "v3-LR",
                "model_version": "v3",
                "configuration": "LR",
                "experiment": "historical+ssp245",
                "member": f"en{member:02d}",
                "variant": "standard",
            }
            for member in range(1, 25)
        },
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["historical", "ssp245"],
        "input_source": "cvdp_rgd",
        "use_pcmdi_model_input": True,
        "control_case_key": "v3-LR-PiControl",
        "control_experiment": "piControl",
        "control_member": "en00",
        "control_period": "v3_lr_piControl",
        "control_case_metadata_by_case": {
            "v3-LR-PiControl": {
                "model_name": "v3.LR.piControl",
                "case_name": "v3-LR",
                "model_version": "v3",
                "configuration": "LR",
                "experiment": "piControl",
                "member": "en00",
                "variant": "standard",
            },
            "v3-LR-PiControl-Spinup": {
                "model_name": "v3.LR.piControl-spinup",
                "case_name": "v3-LR",
                "model_version": "v3",
                "configuration": "LR",
                "experiment": "piControl-spinup",
                "member": "en00",
                "variant": "standard",
            },
        },
        "extra_control_runs": [
            {
                "case_key": "v3-LR-PiControl-Spinup",
                "experiment": "piControl-spinup",
                "member": "en00",
                "period": "v3_lr_piControl_spinup",
            }
        ],
        "amip_cases": [f"v3-LR-AMIP-en{member:02d}" for member in range(1, 4)],
        "amip_period": "amip",
        "cvdp_rgd_root": "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/CVDP_RGD",
        "cvdp_rgd_experiment_dirs": {
            "historical": "v3.LR.historical-SSP245",
            "ssp245": "v3.LR.historical-SSP245",
            "piControl": "v3.LR.piControl",
            "piControl-spinup": "v3.LR.piControl-spinup",
            "amip": "v3.LR.amip",
        },
        "cvdp_rgd_file_prefix_by_experiment": {
            "historical": "v3.LR.historical",
            "ssp245": "v3.LR.historical",
            "piControl": "v3.LR.piControl",
            "piControl-spinup": "v3.LR.piControl-spinup",
            "amip": "v3.LR.amip",
        },
        "clim_mips": ["analysis", "e3sm"],
        "clim_exps": ["historical", "ssp245", "piControl", "piControl-spinup", "amip"],
        "clim_test_mips": ["e3sm"],
        "clim_test_exps": ["historical"],
    },
    "v3_LR_amip": {
        "description": "E3SM v3 LR AMIP ensemble",
        "case_keys": [f"v3-LR-AMIP-en{member:02d}" for member in range(1, 4)],
        "member_alias_by_case": {
            f"v3-LR-AMIP-en{member:02d}": f"en{member:02d}"
            for member in range(1, 4)
        },
        "case_metadata_by_case": {
            f"v3-LR-AMIP-en{member:02d}": {
                "model_name": "v3.LR.amip",
                "case_name": "v3-LR-AMIP",
                "model_version": "v3",
                "configuration": "LR",
                "experiment": "amip",
                "member": f"en{member:02d}",
                "variant": "standard",
            }
            for member in range(1, 4)
        },
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["amip"],
        "input_source": "cvdp_rgd",
        "use_pcmdi_model_input": True,
        "control_case_key": None,
        "control_experiment": None,
        "control_member": None,
        "control_period": None,
        "cvdp_rgd_root": "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/CVDP_RGD",
        "cvdp_rgd_experiment_dirs": {
            "amip": "v3.LR.amip",
        },
        "clim_mips": ["analysis", "e3sm"],
        "clim_exps": ["historical", "amip"],
        "clim_test_mips": ["e3sm"],
        "clim_test_exps": ["amip"],
    },
    "cmip6_historical": {
        "description": "CMIP6 historical multi-model ensemble from CVDP_RGD",
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["historical"],
        "input_source": "cvdp_rgd",
        "use_pcmdi_model_input": True,
        "model_mip": "cmip6",
        "preview_case_limit": 12,
        "control_case_key": None,
        "control_experiment": None,
        "control_member": None,
        "control_period": None,
        "cvdp_rgd_root": "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/CVDP_RGD",
        "cvdp_rgd_experiment_dirs": {
            "historical": "CMIP6_MME_LTM/historical",
        },
        "cvdp_rgd_model_subdir_template": "{case_name}",
        "cvdp_rgd_file_pattern": "{case_name}.{experiment}.{member}.{variable}.*.nc",
        "discover_cvdp_cases": True,
        "discover_experiments": ["historical"],
        "discover_variable": "PSL",
        "discover_primary_member_prefixes": ["r1i1p1f1"],
        "discover_include_all_members": False,
        "clim_mips": ["analysis", "cmip6"],
        "clim_exps": ["historical"],
        "clim_test_mips": ["cmip6"],
        "clim_test_exps": ["historical"],
    },
    "cmip6_amip": {
        "description": "CMIP6 AMIP multi-model ensemble from CVDP_RGD",
        "analysis_output": "atm_ts",
        "analysis_variable": "PSL",
        "workflow_periods": ["cmip6_amip"],
        "experiment_by_period": {
            "cmip6_amip": "amip",
        },
        "lead_lag_period": "cmip6_amip",
        "input_source": "cvdp_rgd",
        "use_pcmdi_model_input": True,
        "model_mip": "cmip6",
        "preview_case_limit": 12,
        "control_case_key": None,
        "control_experiment": None,
        "control_member": None,
        "control_period": None,
        "cvdp_rgd_root": "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/CVDP_RGD",
        "cvdp_rgd_experiment_dirs": {
            "amip": "CMIP6_MME_LTM/amip",
        },
        "cvdp_rgd_model_subdir_template": "{case_name}",
        "cvdp_rgd_file_pattern": "{case_name}.{experiment}.{member}.{variable}.*.nc",
        "discover_cvdp_cases": True,
        "discover_experiments": ["amip"],
        "discover_variable": "PSL",
        "discover_primary_member_prefixes": ["en00"],
        "discover_include_all_members": False,
        "clim_mips": ["analysis", "cmip6"],
        "clim_exps": ["historical", "amip"],
        "clim_test_mips": ["cmip6"],
        "clim_test_exps": ["amip"],
    },
}

MODEL_SUITES["cmip6_historical_all_members"] = copy.deepcopy(MODEL_SUITES["cmip6_historical"])
MODEL_SUITES["cmip6_historical_all_members"].update(
    {
        "description": "CMIP6 historical multi-model ensemble from CVDP_RGD, all available members",
        "discover_include_all_members": True,
    }
)

MODEL_SUITES["cmip6_amip_all_members"] = copy.deepcopy(MODEL_SUITES["cmip6_amip"])
MODEL_SUITES["cmip6_amip_all_members"].update(
    {
        "description": "CMIP6 AMIP multi-model ensemble from CVDP_RGD, all available members",
        "discover_include_all_members": True,
    }
)


OBS_CONFIG = {
    "names": ["NOAA_20C", "ERA5"],
    "experiment": "historical",
    "member": "en00",
    "period_by_name": {
        "NOAA_20C": "195001-201412",
        "ERA5": "197901-201412",
    },
    "variable_by_name": {
        "NOAA_20C": "PSL",
        "ERA5": "psl",
    },
    "clim_sets": ["NOAA_20C.en00", "ERA5.en00"],
    "clim_mips": ["analysis", "analysis"],
    "clim_periods": ["1950-2014", "1979-2014"],
}


def load_suite_config(suite_key, select_model_cases, cmip6_member_selection=None):
    if suite_key not in MODEL_SUITES:
        valid = ", ".join(sorted(MODEL_SUITES))
        raise KeyError(f"Unknown ASL model suite {suite_key!r}. Valid suites: {valid}")

    suite = copy.deepcopy(MODEL_SUITES[suite_key])
    suite["key"] = suite_key
    apply_cmip6_member_selection(suite, cmip6_member_selection)

    if suite.get("discover_cvdp_cases", False):
        case_keys, case_metadata = discover_cvdp_rgd_cases(suite)
        suite["case_keys"] = case_keys
        suite["case_metadata_by_case"] = case_metadata

    if "case_keys" in suite:
        suite["selected_cases"] = list(suite["case_keys"])
    else:
        suite["selected_cases"] = select_model_cases(**suite["selection"])
    return suite


def apply_cmip6_member_selection(suite, cmip6_member_selection):
    if suite.get("model_mip") != "cmip6" or cmip6_member_selection is None:
        return

    selection = str(cmip6_member_selection).strip()
    if not selection:
        return

    if selection.lower() == "all":
        suite["discover_include_all_members"] = True
        return

    suite["discover_include_all_members"] = False
    suite["discover_primary_member_prefixes"] = [selection]


def discover_cvdp_rgd_cases(suite):
    case_keys = []
    case_metadata = {}
    root = suite["cvdp_rgd_root"]
    variable = suite.get("discover_variable", suite["analysis_variable"]).upper()
    primary_member_prefixes = suite.get("discover_primary_member_prefixes", [])
    include_all_members = suite.get("discover_include_all_members", True)

    for experiment in suite["discover_experiments"]:
        exp_dir = suite["cvdp_rgd_experiment_dirs"][experiment]
        search_root = os.path.join(root, exp_dir)
        for model_dir in sorted(glob.glob(os.path.join(search_root, "*"))):
            if not os.path.isdir(model_dir):
                continue

            model_name = os.path.basename(model_dir)
            pattern = os.path.join(
                model_dir,
                f"{model_name}.{experiment}.*.{variable}.*.nc",
            )
            discovered_members = []
            for path in sorted(glob.glob(pattern)):
                basename = os.path.basename(path)
                prefix = f"{model_name}.{experiment}."
                suffix = f".{variable}."
                if not basename.startswith(prefix) or suffix not in basename:
                    continue

                member = basename[len(prefix):basename.index(suffix)]
                discovered_members.append((member, path))

            if not include_all_members and primary_member_prefixes:
                primary_members = [
                    item for item in discovered_members
                    if any(item[0].startswith(prefix) for prefix in primary_member_prefixes)
                ]
                discovered_members = primary_members[:1] or discovered_members[:1]

            for member, _ in discovered_members:
                add_cvdp_case(case_keys, case_metadata, experiment, model_name, member)

    return case_keys, case_metadata


def add_cvdp_case(case_keys, case_metadata, experiment, model_name, member):
    case_key = f"CMIP6-{experiment}-{model_name}-{member}"
    case_keys.append(case_key)
    case_metadata[case_key] = {
        "model_name": f"{model_name}.{experiment}.{member}",
        "case_name": model_name,
        "model_version": "CMIP6",
        "configuration": "MME",
        "experiment": experiment,
        "member": member,
        "variant": "standard",
    }


def format_ym_span(start_ym, end_ym):
    return f"{start_ym:06d}-{end_ym:06d}"


def raw_index_file(out_path, mip, exp, case, relm, case_id, variable, period):
    return os.path.join(
        out_path,
        f"{mip}.{exp}.{case}.{relm}.{case_id}.{variable.upper()}.{period}.csv",
    )


def lead_lag_file(out_path, reg_idx, mip, exp, case, relm, case_id, variable, period):
    return os.path.join(
        out_path,
        f"{reg_idx}_leadlag_{mip}.{exp}.{case}.{relm}.{case_id}.{variable}.{period}.nc",
    )


def should_run(path, force_recompute=False, label="output"):
    if os.path.exists(path) and not force_recompute:
        print(f"Skip existing {label}: {path}")
        return False
    return True


def print_workflow_switches(
    active_model_suite,
    process_observations,
    process_model_indices,
    process_control_index,
    process_climatology,
    process_lead_lag,
    force_recompute,
):
    print(f"Active model suite      : {active_model_suite}")
    print(f"Process observations    : {process_observations}")
    print(f"Process model indices   : {process_model_indices}")
    print(f"Process control index   : {process_control_index}")
    print(f"Process climatology     : {process_climatology}")
    print(f"Process lead-lag        : {process_lead_lag}")
    print(f"Force recompute         : {force_recompute}")
