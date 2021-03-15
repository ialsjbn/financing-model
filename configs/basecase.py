#### GENERAL INPUT ####
# Input files
building_input = 'data/building_damage_owner.pkl' # don't change

# Filename to output
filename_output = 'results/basecase.csv' # change this

# Choose scenario number
scenario_num = 1

# hazus cost percentage per damage level
cost_pct = [0, 0.002, 0.1, 0.5, 1] 


#### FINANCING INPUTS ####
# Area median income
ami = 104234

# permitting 
perm_small = 193+927+1490
perm_med = 386+1590+2168
perm_large = 579+2385+3116

# Insurance
insurance_penetration = 0.13 # California Insurance Authority
insurance_deduct_pct = 0.15 
ins_med = 6*7 # REDi
ins_beta = 1.11 # REDi

# Bank loan
bank_med = 60
bank_beta = 0.68
bank_approval_high = 0.91
bank_approval_mod = 0.58
bank_approval_low = 0.19
bank_amount_pct = 0.5

# FEMA-IHP grant
fema_approval_mod = 0.46
fema_approval_low = 0.51
fema_approval_verylow = 0.63

fema_repair_med = 2513.22 # OpenFEMA dataset
fema_repair_beta = 1.37 # OpenFEMA dataset
fema_replace_a1 = 237390.6 # OpenFEMA dataset
fema_replace_b1 = 45.57 # OpenFEMA dataset
fema_replace_loc1 = -172059279.3 # OpenFEMA dataset
fema_replace_scale1 = 172121361.75 # OpenFEMA dataset
fema_max = 34000
fema_med = 80 # GAO report
fema_beta = 0.57 # REDi

# SBA
sba_approval_rate = 0.4768 # SBA open data
sba_thres = 30750
sba_coef = 0.449 # SBA open data
sba_max = 200000
sba_med = 45 # SBA report
sba_beta = 0.57 # REDi

# CDBG-DR
cdbg_pool_pct = 0.1*0.4
# cdbg_pool = 640000000 # CDBG-DR CA 2018
cdbg_max = 150000
cdbg_start = 1.5*365
cdbg_end = 6*365

# NGO
# ngo_pool = 200000000
ngo_repair_pct = 0.1
ngo_rebuild_pct = 0.05
ngo_med = 3*30
ngo_beta = 1.11

# Personal Savings
savings_pct = 0.5

# Parametric Payout
param_payout = False

#### RECOVERY INPUT #####
# HAZUS parameters
cons_time = [0, 2, 30, 90, 180]
beta = 0.4

# Simulation parameters
SIM_TIME = 5000
NUM_CONTRACTORS = 1200
split_resource = False