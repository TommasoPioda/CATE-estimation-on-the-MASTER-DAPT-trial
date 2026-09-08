index_cols = [
    'randomized_pop',
    'mnppid',
    'mnpaid',
    'mnpctrid', 
    'mnpctrname',
    'centre',
    'mnpcname',
    'mnp_regimen_gr'
]

target_cols = [
    'cec_barc235_335d',
    'cec_cvdeath_335d',
    'cec_mi_335d',
    'cec_stroke_335d',
    'cec_bleed_335d',
    'cec_barc2_335d',
    'cec_barc3_335d',
    'cec_barc35_335d',
    'cec_revasc_335d',
]

binary = [
    "fup_other",
    "rand_oac12",
    "cancer",
    "prec_priorbleed",
    "p_bleed",
    "rand_ster_nsaid",
    "rand_anemia",
    "fup_nitrates",
    "fup_insulin",
    "prior_cabg",
    "prior_mi_12m",
    "cad",
    "hyperlipidemia",
    "regimen"
]

numeric = [
    "pci_tropo_pre",
    "ic_age_mon",
    "pci_platel_pre",
    "fup_nyha",
    "lvef",
    "heartrate",
    "pci_glycemia",
    "prec_hb",
    "pci_creatinine_pre",
    "prec_wbc",
    "bmi",
    "pci_platelvol_pre",
    "prior_mi",
    "bp_sys",
    "contrast",
    "pci_platelvol_ldis",
    "prior_pci",
    "prec_creatinine",
    "pci_tropo_ldis",
    "height",
    "fup_ccs",
]

categorical = [
    "afib",
    "alcohol",
    "diabetes",
    "geography",
    "p_bleed_action1",
    "proir_avs_yn",
    "acs_last_pci",
    "acs",
    "smoker",
    "gender",
    "race",
]