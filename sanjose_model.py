import sys
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm
import random
import simpy
from tqdm import tqdm
from configs.basecase import * # Change this according to parameters

def dollars_needed():
	'''
	Function to calculate dollars needed given damage
	'''
	global hh

	# Calculate dollars needed to reconstruct
	for i in tqdm(range(len(hh))):
	    area = hh.loc[i, 'SQUARE_FOO']
	    ds = hh.loc[i, 'DAMAGE']
	    bldg_value = hh.loc[i, 'IMPROVEMEN']
	    
	    # # Permitting cost
	    # if area < 750:
	    #     perm_cost = perm_small
	    # elif area < 2250:
	    #     perm_cost = perm_med
	    # else:
	    #     perm_cost = perm_large
	    perm_cost = 0
	    
	    # Total dollar needed
	    hh.loc[i, 'dollar_needed'] = cost_pct[ds]*bldg_value + perm_cost

	# Calculate Balance
	hh['balance'] = hh['dollar_needed']
	hh['home_ins'] = hh[['IMPROVEMEN', 'home_ins']].min(axis = 1) # Makes sure insurance is not greater than home value

def earthquake_insurance():
	'''
	Function to assign insurance to households
	'''
	global hh

	print('Calculating Insurance')
	# Randomly sample 13% of households to get insurance
	ind_ins = random.sample(range(len(hh)), int(len(hh)* insurance_penetration))

	# Assign insurance
	# count = 0
	for i in tqdm(range(len(ind_ins))):
	    
	    if hh.loc[ind_ins[i],'dollar_needed'] > insurance_deduct_pct*hh.loc[ind_ins[i], 'IMPROVEMEN']:
	        hh.loc[ind_ins[i],'insurance'] = min(hh.loc[ind_ins[i],'IMPROVEMEN'], hh.loc[ind_ins[i],'dollar_needed'])- insurance_deduct_pct*hh.loc[ind_ins[i], 'IMPROVEMEN']
	        hh.loc[ind_ins[i],'insurance_time'] = np.random.lognormal(np.log(ins_med), ins_beta)
	        hh.loc[ind_ins[i],'balance'] -= hh.loc[ind_ins[i], 'insurance']
	        # count += 1

	print('Number of Households w/ Insurance:', len(hh.loc[hh['insurance'] > 0]))
	# return hh

def fema_ihp(ds, balance, fema_approval_rate):
    '''
    Function to calculate FEMA IHP aid received for a single house

    Input: 
    ds: damage level (1-3)
    balance: remaining dollars needed to reconstruct
	fema_approval_rate: approval rate of the household given income

    Output: 
    fema_ihp: amount of aid received
    fema_time: time to receive FEMA IHP
    '''
    prob = random.random()
    fema_time = 0
    fema_ihp = 0
    
    if prob < fema_approval_rate and balance > 0: # eligible for FEMA IHP
        if ds == 4:
            # replacement funds
            fema_fund = min(scipy.stats.beta.rvs(fema_replace_a1,fema_replace_b1,
                                                 fema_replace_loc1,fema_replace_scale1),
                           fema_max)
            fema_ihp = min((fema_fund), balance)
            fema_time = np.random.lognormal(np.log(fema_med), fema_beta)

        elif fema_approval_rate == 1 and (ds == 3 or ds == 2): # intervention 1
            fema_fund = min(np.random.lognormal(np.log(fema_repair_med), fema_repair_beta), fema_max)
            fema_ihp = min((fema_fund), balance)
            fema_time = np.random.lognormal(np.log(fema_med), fema_beta)

        elif fema_approval_rate < 1 and ds == 3:
            fema_fund = min(np.random.lognormal(np.log(fema_repair_med), fema_repair_beta), fema_max)
            fema_ihp = min((fema_fund), balance)
            fema_time = np.random.lognormal(np.log(fema_med), fema_beta)
            
    return fema_ihp, fema_time

def bank_loan(balance, income, bank_amount_pct, bank_approval_rate):
    '''
    Function to calculate amount of bank loan received for a single house

    Input: 
    balance: remaining dollars needed to reconstruct
    income: household income
    bank_amount_pct: debt-to-income ratio
    bank_approval_rate: approval rate to get bank loan given income

    Output: 
    bank: amount of loan received
    bank_time: time to receive bank loan
    '''
    prob = random.random()
    bank_time = 0
    bank = 0
    
    if prob < bank_approval_rate and balance > 0:
        bank = min(balance, bank_amount_pct*income)
        bank_time = np.random.lognormal(np.log(bank_med), bank_beta)
        
    return bank, bank_time

def sba(ds, balance):
    '''
    Function to calculate SBA loan received for a single house

    Input:
    ds: damage level (1-3)
    balance: remaining dollars needed to reconstruct

    Output: 
    sba: amount of aid received
    sba_time: time to receive FEMA IHP
    '''
    prob = random.random()
    sba_time = 0
    sba = 0
    
    if prob < sba_approval_rate and balance > 0: # eligible for SBA
        sba_fund = min(sba_coef * balance, sba_max)
        sba = min(sba_fund, balance)

        sba_time = np.random.lognormal(np.log(sba_med), sba_beta)
            
    return sba, sba_time

def financing_model(ngo_repair_pct, ngo_rebuild_pct, cdbg_pool_pct):
	'''
	Function to run the financing model

	Input: 
	ngo_repair_pct: % of houses able to be repaired through NGO
	ngo_rebuild_pct: % of houses able to be rebuilt through NGO
	cdbg_pool_pct: Amount of CDBG funds available in terms of % of total damage loss
	'''
	global hh

	# Define income groups
	high_inc = hh['income_indiv'] >= 2*ami
	mod_inc = (hh['income_indiv'] > 0.8*ami) & (hh['income_indiv'] < 2* ami)
	low_inc = (hh['income_indiv'] > 0.5*ami) & (hh['income_indiv'] <= 0.8* ami)
	very_low_inc = hh['income_indiv'] <= 0.5*ami

	# Assign EQ insurance for all
	earthquake_insurance()

	# High income
	print('Calculating High Income')
	# Bank loan
	high_ind = hh[high_inc].index.tolist()
	for i in tqdm(high_ind):
		hh.loc[i,'bank'], hh.loc[i,'bank_time'] = bank_loan(hh.loc[i,'balance'], hh.loc[i, 'income_indiv'], bank_amount_pct, bank_approval_high)
		hh.loc[i,'balance'] -= hh.loc[i,'bank']

	# Moderate Income
	print('Calculating Moderate Income')
	mod_ind = hh[mod_inc].index.tolist() 	# Get rows with moderate income
	for i in tqdm(mod_ind):
	    if hh.loc[i,'HOME_OWNER'] == 'H':

	        # SBA
	        hh.loc[i,'sba'], hh.loc[i,'sba_time'] = sba(hh.loc[i,'DAMAGE'], hh.loc[i,'balance'])
	        hh.loc[i,'balance'] -= hh.loc[i,'sba']
	        
	        
	        if hh.loc[i, 'sba'] == 0: # if didn't get SBA
	        	# Bank loan: still might not get it
	        	hh.loc[i,'bank'], hh.loc[i,'bank_time'] = bank_loan(hh.loc[i,'balance'], hh.loc[i, 'income_indiv'], bank_amount_pct, bank_approval_mod)

	        	# FEMA-IHP
		        hh.loc[i,'fema'], hh.loc[i,'fema_time'] = fema_ihp(hh.loc[i,'DAMAGE'], hh.loc[i,'balance'],fema_approval_mod)
		        hh.loc[i,'balance'] -= hh.loc[i,'fema']

        	else: # if they already got SBA, they for sure can get bank loan

        		hh.loc[i, 'bank'] = np.minimum(hh.loc[i, 'balance'], bank_amount_pct * hh.loc[i, 'income_indiv'])
        		hh.loc[i, 'bank_time'] = np.random.lognormal(np.log(bank_med), bank_beta)
	        hh.loc[i,'balance'] -= hh.loc[i,'bank']

	print('Calculating Low Income')
	# Low income
	low_ind = hh[low_inc].index.tolist() # Get rows with low income
	for i in tqdm(low_ind):
	    if hh.loc[i,'HOME_OWNER'] == 'H':
    
	        # SBA
	        hh.loc[i,'sba'], hh.loc[i,'sba_time'] = sba(hh.loc[i,'DAMAGE'], hh.loc[i,'balance'])
	        hh.loc[i,'balance'] -= hh.loc[i,'sba']
	        
	        if hh.loc[i, 'sba'] == 0: # if didn't get SBA
	        	# Bank loan: still might not get it
	        	hh.loc[i,'bank'], hh.loc[i,'bank_time'] = bank_loan(hh.loc[i,'balance'], hh.loc[i, 'income_indiv'], bank_amount_pct, bank_approval_low)

	        	# FEMA-IHP
		        hh.loc[i,'fema'], hh.loc[i,'fema_time'] = fema_ihp(hh.loc[i,'DAMAGE'], hh.loc[i,'balance'], fema_approval_low)
		        hh.loc[i,'balance'] -= hh.loc[i,'fema']

        	else: # if they already got SBA, they for sure can get bank loan

        		hh.loc[i, 'bank'] = np.minimum(hh.loc[i, 'balance'], bank_amount_pct * hh.loc[i, 'income_indiv'])
        		hh.loc[i, 'bank_time'] = np.random.lognormal(np.log(bank_med), bank_beta)

	        hh.loc[i,'balance'] -= hh.loc[i,'bank']

	print('Calculating Very Low Income')
	# Very Low income
	very_low_ind = hh[very_low_inc].index.tolist() # Get rows with very low income
	for i in tqdm(very_low_ind):
	    if hh.loc[i,'HOME_OWNER'] == 'H':
    
	        # FEMA-IHP
	        hh.loc[i,'fema'], hh.loc[i,'fema_time'] = fema_ihp(hh.loc[i,'DAMAGE'], hh.loc[i,'balance'], fema_approval_verylow)
	        hh.loc[i,'balance'] -= hh.loc[i,'fema']	

	# NGO
	print('Calculating NGO')

	# Calculate number of houses that can be helped
	total_ds34 = sum(hh['DAMAGE'] == 3) + sum(hh['DAMAGE'] == 4)
	ngo_ds3 = round(total_ds34*ngo_repair_pct)
	ngo_ds4 = round(total_ds34*ngo_rebuild_pct)
	print('Number of possible NGO repair', ngo_ds3)
	print('Number of possible NGO rebuild', ngo_ds4)

	# Get low and very low income homeowners that still needs money
	low_inc_need_3 = low_inc & (hh['balance'] > 0) & (hh['HOME_OWNER']  == 'H') & (hh['DAMAGE']  <= 3)
	low_ind_need_3 = hh[low_inc_need_3].index.tolist()

	very_low_inc_need_3 = very_low_inc & (hh['balance'] > 0) & (hh['HOME_OWNER']  == 'H') & (hh['DAMAGE']  <= 3)
	very_low_ind_need_3 = hh[very_low_inc_need_3].index.tolist()

	low_inc_need_4 = low_inc & (hh['balance'] > 0) & (hh['HOME_OWNER']  == 'H') & (hh['DAMAGE']  == 4)
	low_ind_need_4 = hh[low_inc_need_4].index.tolist()

	very_low_inc_need_4 = very_low_inc & (hh['balance'] > 0) & (hh['HOME_OWNER']  == 'H') & (hh['DAMAGE']  == 4)
	very_low_ind_need_4 = hh[very_low_inc_need_4].index.tolist()

	# repair
	count_ngo_3 = 0
	random.shuffle(very_low_ind_need_3)
	random.shuffle(low_ind_need_3)
	repair_queue =  very_low_ind_need_3 + low_ind_need_3
	for i in repair_queue:
	    if count_ngo_3 < ngo_ds3:
	        hh.loc[i, 'ngo'] = hh.loc[i, 'balance']
	        hh.loc[i, 'ngo_time'] = np.random.lognormal(np.log(ngo_med), ngo_beta)
	        hh.loc[i, 'balance'] -= hh.loc[i, 'ngo']
	        count_ngo_3 += 1
	    else:
	        break

	# rebuild
	count_ngo_4 = 0
	random.shuffle(very_low_ind_need_4)
	random.shuffle(low_ind_need_4)
	repair_queue =  very_low_ind_need_4 + low_ind_need_4
	for i in repair_queue:
	    if count_ngo_4 < ngo_ds4:
	        hh.loc[i, 'ngo'] = hh.loc[i, 'balance']
	        hh.loc[i, 'ngo_time'] = np.random.lognormal(np.log(ngo_med), ngo_beta)
	        hh.loc[i, 'balance'] -= hh.loc[i, 'ngo']
	        count_ngo_4 += 1
	    else:
	        break

	print('Number of NGO beneficiaries (repair):', str(count_ngo_3))
	print('Number of NGO beneficiaries (rebuild):', str(count_ngo_4))

	# CDBGR
	print('Calculating CDBGR')
	# Assign design code
	conditions = [
	    (hh['income_indiv'] <= 0.8*ami) & (hh['DAMAGE'] >= 3) & (hh['HOME_OWNER'] == 'H'),
	    (hh['income_indiv'] <= 0.8*ami) & (hh['DAMAGE'] < 3) & (hh['HOME_OWNER'] == 'H'),
	    (hh['income_indiv'] > 0.8*ami) & (hh['DAMAGE'] >= 3) & (hh['HOME_OWNER'] == 'H'),
	    (hh['income_indiv'] > 0.8*ami) & (hh['DAMAGE'] < 3) & (hh['HOME_OWNER'] == 'H'),
	    (hh['HOME_OWNER'] != 'H')
	]

	# design code values
	values = [1,2,3,4,5]

	# create new colum and use np.select to assign values
	hh['cdbg_tier'] = np.select(conditions, values)

	# separate dataframe
	data_queue = [hh[(hh['cdbg_tier'] == 1) & (hh['balance'] > 0)],
	              hh[(hh['cdbg_tier'] == 2) & (hh['balance'] > 0)],
	              hh[(hh['cdbg_tier'] == 3) & (hh['balance'] > 0)],
	              hh[(hh['cdbg_tier'] == 4) & (hh['balance'] > 0)]]

	# calculate available CDBG-DR fund
	cdbg_pool = cdbg_pool_pct * sum(hh['dollar_needed'])
	print('Available CDBG fund: ', cdbg_pool)

	# count_cdbg = 0
	for tier_data in data_queue:
	    idx = tier_data.index.values
	    random.shuffle(idx)
	    for i in idx:
	        if cdbg_pool > 0:
	            hh.loc[i,'cdbg'] = min(cdbg_max, hh.loc[i,'balance'], cdbg_pool)
	            hh.loc[i,'cdbg_time'] = np.random.uniform(cdbg_start, cdbg_end)
	            hh.loc[i,'balance'] -= hh.loc[i,'cdbg']
	            cdbg_pool -= hh.loc[i,'cdbg']
	            # count_cdbg += 1
	        else:
	            break

	print('Number of CDBG beneficiaries:', len(hh.loc[hh['cdbg'] > 0]))

	# Intervention 3: Parametric Payout
	if param_payout:
		# hh['param'] = np.where((hh['balance'] <= param_amount) & very_low_inc, hh['balance'], 0)
		# hh['balance'] -= hh['param']
		# hh.loc[hh['param'] > 0, 'param_time'] = np.random.lognormal(np.log(ins_med), ins_beta, len(hh['param'] >0))

		hh.loc[very_low_inc, 'param'] = np.minimum(hh.loc[very_low_inc, 'balance'].values, param_amount)
		hh.loc[very_low_inc, 'balance'] = hh.loc[very_low_inc, 'balance'] - hh.loc[very_low_inc, 'param']
		hh.loc[very_low_inc, 'param_time'] = np.random.lognormal(np.log(bank_med), bank_beta, np.count_nonzero(very_low_inc))

	# Personal Savings
	print('Calculating Personal Savings')
	# assume that if the leftover dollar needed is <= 50% of median income, they can reconstruct
	hh['savings'] = np.where((hh['balance'] <= savings_pct*hh['income_indiv']) & (hh['income_indiv'] > 0.5*ami), hh['balance'], 0)
	hh['balance'] -= hh['savings']

	# Count total time
	# High income
	hh.loc[high_inc, 'total_time'] = pd.concat([hh.loc[high_inc,['insurance_time', 'bank_time']].sum(axis = 1),hh.loc[high_inc, 'cdbg_time']], axis = 1).max(axis = 1)

	# Moderate income
	tmp1 = hh.loc[mod_inc,['insurance_time', 'fema_time']].max(axis = 1) 
	tmp2 = tmp1 + hh.loc[mod_inc, 'sba_time'] + hh.loc[mod_inc, 'bank_time']
	hh.loc[mod_inc, 'total_time'] = pd.concat([tmp2, hh.loc[mod_inc, 'cdbg_time']], axis = 1).max(axis = 1)
	# hh.loc[mod_inc, 'insurance_time'] + hh.loc[mod_inc, 'fema_time'] + hh.loc[mod_inc, 'sba_time'] + hh.loc[mod_inc, 'bank_time'] + hh.loc[mod_inc, 'cdbg_time']

	# Low income
	tmp1 = hh.loc[low_inc,['insurance_time', 'fema_time']].max(axis = 1) 
	tmp2 = tmp1 + hh.loc[low_inc, 'sba_time'] + hh.loc[low_inc, 'bank_time'] + hh.loc[low_inc, 'ngo_time']
	hh.loc[low_inc, 'total_time'] = pd.concat([tmp2, hh.loc[low_inc, 'cdbg_time']], axis = 1).max(axis = 1)
	# hh.loc[low_inc, 'total_time'] = hh.loc[low_inc, 'insurance_time'] + hh.loc[low_inc, 'fema_time'] + hh.loc[low_inc, 'sba_time'] + hh.loc[low_inc, 'ngo_time'] + hh.loc[low_inc, 'cdbg_time']

	# Very low income
	tmp1 = hh.loc[very_low_inc,['insurance_time', 'param_time', 'fema_time']].max(axis = 1) 
	tmp2 = tmp1 + hh.loc[very_low_inc, 'ngo_time']
	hh.loc[very_low_inc, 'total_time'] = pd.concat([tmp2, hh.loc[very_low_inc, 'cdbg_time']], axis = 1).max(axis = 1)

class Region(object):
    """A region has a limited number of contractors (``NUM_CONTRACTOR``) to
    construct buildings in parallel.
    
    Buildings have to request a contractor to build their house. When they
    get one, they can start the rebuilding process and wait for it to finish
    (which takes 'cons_time' days)

    """
    def __init__(self, env, num_contractor):
        self.env = env
        self.contractor = simpy.Resource(env, num_contractor)

    def rebuild(self, building, cons_time):
        """The rebuilding process. It takes a ``building`` process and tries
        to rebuild it"""
        yield self.env.timeout(cons_time)

    def delay(self, building, fin_time):
        """The financing time"""
        yield self.env.timeout(fin_time)
        

def building(env, bldg_id, rg, damage, fin_time, cons_time, data):
    """The building (each building has a ``bldg_id`` and 
    damage level 'damage') arrives at the region (``rg``) after waiting to
    receive financing fin_time and requests a rebuild.
    
    It then starts the rebuilding process, which takes a cons_time
    that is lognormally distributed. waits for it to finish and
    is reconstructured. 

    """
    
    # Only damage 3 and 4 compete for resources
    if damage >=3:
        # Financing time
        # print('Building %d starts financing at %.2f.' % (bldg_id, env.now))
        yield env.process(rg.delay(bldg_id, fin_time))
        # print('Building %d finishes financing at %.2f.' % (bldg_id, env.now))
        
        # Construction
        with rg.contractor.request() as request:
            yield request

            # Construction
            # print('Building %d with damage level %s starts construction at %.2f.' % (bldg_id, str(damage), env.now))
            start_cons_time = env.now

            yield env.process(rg.rebuild(bldg_id, cons_time))

            # print('Building %d with damage level %s finishes construction at %.2f took %.2f days' % (bldg_id,str(damage), env.now, cons_time))

            # Append data of construction times
            data.append((bldg_id, damage, start_cons_time, env.now))
    else:
        # don't compete for resources
        data.append((bldg_id, damage, fin_time, fin_time + cons_time))

def setup(env, num_contractor, damage_building, bldg_id, fin_time, cons_time, data):
    """Create a region and number of damaged buildings"""
    
    # Create the region
    region = Region(env, num_contractor)

    # Create buildings initially
    for i in range(len(damage_building)):
        env.process(building(env, bldg_id[i], region, 
                             damage_building[i], fin_time[i], cons_time[i], data))
        yield env.timeout(0)

def recovery_model(cons_time, split_resource):
	'''
	Function to run the recovery model with limited number of contractors and saves
	the result in a csv file.

	Input: 
	cons_time: HAZUS parameters for construction time based on damage level
	split_resource: indicator to implement Intervention 2
	'''
	global hh

	# Simulate construction time
	cons_time = np.array(cons_time)
	hh['cons_time'] = np.random.lognormal(np.log(cons_time[hh['DAMAGE']]), beta)

	if split_resource:
		# Moderate and High Income
		mask = ((hh['balance'] == 0) & (hh['income_indiv'] > 0.8*ami))
		DAMAGE_BUILDING = hh.loc[mask, 'DAMAGE'].values
		FIN_TIME = hh.loc[mask, 'total_time'].values
		CONS_TIME = hh.loc[mask, 'cons_time'].values
		BLDG_ID = hh.loc[mask, 'OBJECTID'].values
		num_contractor_high = round(pct_cons_high*NUM_CONTRACTORS)

		# Setup and start the simulation
		data_comp_high = []
		enable_print = 0
		random.seed(0)  # This helps reproduce the results
		env = simpy.Environment()
		env.process(setup(env, num_contractor_high, DAMAGE_BUILDING, BLDG_ID, FIN_TIME, CONS_TIME, data_comp_high))
		env.run(until=SIM_TIME)

		# Low and Very Low Income
		mask = ((hh['balance'] == 0) & (hh['income_indiv'] <= 0.8*ami))
		DAMAGE_BUILDING = hh.loc[mask, 'DAMAGE'].values
		FIN_TIME = hh.loc[mask, 'total_time'].values
		CONS_TIME = hh.loc[mask, 'cons_time'].values
		BLDG_ID = hh.loc[mask, 'OBJECTID'].values
		num_contractor_low = round(pct_cons_low*NUM_CONTRACTORS)

		# Setup and start the simulation
		data_comp_low = []
		enable_print = 0
		random.seed(0)  # This helps reproduce the results
		env = simpy.Environment()
		env.process(setup(env, num_contractor_low, DAMAGE_BUILDING, BLDG_ID, FIN_TIME, CONS_TIME, data_comp_low))
		env.run(until=SIM_TIME)

		# Combine results
		recov_df_high = pd.DataFrame(data_comp_high, columns = ['OBJECTID', 'DAMAGE', 'cons_start', 'cons_finish'])
		recov_df_low = pd.DataFrame(data_comp_low, columns = ['OBJECTID', 'DAMAGE', 'cons_start', 'cons_finish'])
		recov_df = pd.concat([recov_df_low, recov_df_high], ignore_index = True, sort = False)
	
	else:
		# Get only buildings with 0 balance
		mask = (hh['balance'] == 0)
		DAMAGE_BUILDING = hh.loc[mask, 'DAMAGE'].values
		FIN_TIME = hh.loc[mask, 'total_time'].values
		CONS_TIME = hh.loc[mask, 'cons_time'].values
		BLDG_ID = hh.loc[mask, 'OBJECTID'].values

		# Setup and start the simulation
		data_comp = []
		enable_print = 0
		random.seed(0)  # This helps reproduce the results

		# Create an environment and start the setup process
		env = simpy.Environment()
		env.process(setup(env, NUM_CONTRACTORS, DAMAGE_BUILDING, BLDG_ID, FIN_TIME, CONS_TIME, data_comp))

		# Execute!
		env.run(until=SIM_TIME)

		# Combine results
		recov_df = pd.DataFrame(data_comp, columns = ['OBJECTID', 'DAMAGE', 'cons_start', 'cons_finish'])
	
	# Combine with original dataframe
	buildings_recov = pd.merge(hh, recov_df[['OBJECTID', 'cons_start', 'cons_finish']], on = 'OBJECTID', how = 'outer')

	# Export results
	buildings_recov.to_csv(filename_output)

if __name__ == '__main__': 
	
	print()
	print('Importing Data and Parameters...')
	print('Configuration file: ', list(sys.modules)[-1])
	print('Output file: ', filename_output)

	# Import building damage data
	bldg_data = pd.read_csv(building_input, index_col = 0)
	print('Number of Homeowner Buildings:', len(bldg_data))
	
	# Create dataframe
	eq_scenario_name = 'damage_' + str(scenario_num)
	print('Earthquake Scenario Number:', scenario_num)
	hh = bldg_data[['OBJECTID', 'IMPROVEMEN', eq_scenario_name, 'HOME_OWNER', 'SQUARE_FOO', 
                'income_indiv', 'home_ins','fin_asset', 'geometry']].copy()
	hh = hh.rename(columns = {eq_scenario_name: 'DAMAGE'}) # rename column

	print('Number of damaged buildings: ', sum(hh['DAMAGE'] > 0))
	hh = hh.loc[hh['DAMAGE'] > 0] # get only buildings that are damaged
	hh = hh.reset_index(drop=True)

	# Add columns to dataframe
	col_names = ['dollar_needed', 'balance', 'insurance', 'fema', 'sba', 'cdbg', 'bank','ngo', 'param',
	'insurance_time', 'fema_time', 'sba_time', 'cdbg_time', 'bank_time', 'ngo_time','param_time', 'total_time']
	d = dict.fromkeys(col_names, 0)
	hh = hh.assign(**d)
	print()

	# Calculate Dollars Needed
	print('Calculating Dollars Needed...')
	dollars_needed()
	print()

	# Run Financing Model
	print('Running Financing Model...')
	financing_model(ngo_repair_pct, ngo_rebuild_pct, cdbg_pool_pct)
	print()

	# Run Recovery Model
	print('Running Recovery Model...')
	recovery_model(cons_time, split_resource)
	print()

	print('Done! File Saved to: ', filename_output)

