# Analysis as done in app:
# 1) Specify houshold composition
# 2) Specify enough budget for given houshold
# 3) Convert to equivalized disposable income with 2 adults, 0 children (as used in hdi dataset)
# Metrics to provide:
# - For % of people in households with enough: Get percentile of equivalized budget from decile look-up table
# - For % of people in households with enough, by household composition: 
#           Also, find cumulative % of each household composition at this percentile
# - For what % of enough do bottom 10%/top 10% have:
#           Ratio of these decile points with equivalized disposable income
#              CAVEAT: only going to be strictly accurate for 2 adult/0 children household
# - For economy growth etc:
#           Funds needed to get everyone to a percentile equivalized budget
#           Funds in excess of a percentile equivalized budget

# So... tables should be:
# A) percentage of ALL households in [equivalized disposable income decile] x [Household composition]
# B) cumulative percentage of ALL households in [equivalized disposable income decile] x [Household composition]
# C) percentage of given household composition in [equivalized disposable income decile] x [Household composition]
# D) cumulative percentage of given household composition in [equivalized disposable income decile] x [Household composition]
# E) income decile (non-equivalized) in [equivalized disposable income decile] x [Household composition]
# F) [What is spent below a decile] = n_households * cumulative(A * E) - accumulate from bottom
# G) [What is spent above a decile] = n_households * cumulative(A * E) - accumulate from top
# H) [What should be incomesum below a decile] = n_households * B * [enough by household composition]
# I) [What should be incomesum above a decile] = n_households * (1 - B) * [enough by household composition]
# J) [Missing] = (H - F)
# K) [Excess] = (G - I)

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

import pickle

INDIR = r'./ons-income-2022/'
filename = INDIR + 'hdi_composition_nonretired.xlsx'


# decile points, in terms of equivalized household income (to 2 adult household)
# by individual (i.e. each decile has equal number of people, not equal number of households)
#This spreadsheet has the decile averages interleaved with the decile edges
#The decile averages are not the decile median values, so this is not quite right, but not far off
sheetname = 'deciles'
deciles20 = pd.read_excel(filename, sheetname, index_col=0, header=0).squeeze("columns")

#need to set min and max to something sensible (PLACEHOLDER in there for now)
deciles20.loc[0] = 2*deciles20.loc[0.5] - deciles20.loc[1]
deciles20.loc[10] = 2*deciles20.loc[9.5] - deciles20.loc[9]
deciles20.index = deciles20.index/10

interp_HEDI = np.asarray(deciles20)
interp_pcInd = np.asarray(deciles20.index)
#add some extremes so we don't get errors...
interp_HEDI = np.insert(interp_HEDI, 0, 0)
interp_pcInd = np.insert(interp_pcInd, 0, -1)
interp_HEDI = np.append(interp_HEDI, 1e12)
interp_pcInd = np.append(interp_pcInd, 1.1)

#This gives us Household equivalized dispoable income to percentile INDIVIDUAL (and vice versa)
f_HEDI_to_pcInd = interp1d(interp_HEDI, interp_pcInd)
f_pcInd_to_HEDI = interp1d(interp_pcInd, interp_HEDI)

#repeat for decile averages only
sheetname = 'deciles_av'
deciles_av = pd.read_excel(filename, sheetname, index_col=0, header=0).squeeze("columns")
deciles_av.index = deciles_av.index/10
l_deciles_av = deciles_av.tolist()

#repeat for decile edges only
sheetname = 'deciles_edges'
deciles_edges = pd.read_excel(filename, sheetname, index_col=0, header=0).squeeze("columns")
deciles_edges.index = deciles_edges.index/10
deciles_edges.loc[0] = deciles20.loc[0]
deciles_edges.loc[1] = deciles20.loc[1]

#Now get the percentages in terms of households, not individuals...

# number of households by decile
n_households = pd.read_excel(filename, 'n_households', index_col=0, header=0).squeeze("columns")
tot_households = n_households.sum()
pc_households = n_households/tot_households
tot_households *= 1000
pc_households = pc_households.cumsum()
pc_households.loc[0] = 0
pc_households.sort_index(inplace=True)
pc_households.index = pc_households.index/10

f_pcInd_to_pcHouse = interp1d(np.asarray(pc_households.index), np.asarray(pc_households))

#print(pc_households)
print(tot_households)

#Now percentage of a given household composition

sheetname = 'composition'
df_comp = pd.read_excel(filename, sheetname, index_col=1, header=0)
df_comp.drop(columns='Unnamed: 0', inplace=True)

#express percentiles as per ALL households, rather than decile composition
df_comp = df_comp/100
for col in df_comp:
    df_comp[col] *= n_households[col] * 1000

df_comp = df_comp.transpose()
df_comp.index = df_comp.index/10
df_comp_pc = df_comp/tot_households
df_comp_pc_bycomp = df_comp/df_comp.sum()

df_comp_pc_bycomp = df_comp_pc_bycomp.cumsum()
df_comp_pc_bycomp.loc[0] = 0
df_comp_pc_bycomp.sort_index(inplace=True)

f_pcInd_to_pcHouse_byComp = interp1d(np.asarray(pc_households.index), np.asarray(df_comp_pc_bycomp), axis=0)

#save household composition labels and column indices
household_comps = df_comp.columns.to_list()
d_household_comps_to_index = {comp : idx for idx, comp in enumerate(household_comps)}

# OECD-modified equivalized - 1 for first adult, 0.5 for subsequent adults, 0.3 for children
first_adult = 1
subs_adults = 0.5
children = 0.3

#can't pickle lambda functions so name these functions
def composition_to_equiv_factor (na, nc):
    return (first_adult + max(na-1, 0)*subs_adults + nc*children)

def equivalize(val, na, nc):
    return val * (composition_to_equiv_factor(2, 0)/composition_to_equiv_factor(na, nc))

def dequivalize(val, na, nc):
    return val * (composition_to_equiv_factor(na, nc)/composition_to_equiv_factor(2, 0))

eff_adults_3_or_more_adults_no_children = 3.5
eff_children_1_adult_with_children = 1.5
eff_children_2_adults_with_3_or_more_children = 3.5
eff_adults_3_or_more_adults_with_children = 3.5
eff_children_3_or_more_adults_with_children = 2

#number of adults and children in each household composition
nas = [1, 2, eff_adults_3_or_more_adults_no_children, 1, 2, 2, 2, eff_adults_3_or_more_adults_with_children]
ncs = [0, 0, 0, eff_children_1_adult_with_children, 1, 2, eff_children_2_adults_with_3_or_more_children, eff_children_3_or_more_adults_with_children]

#dequivalized household income by household composition
data_income = np.asarray([dequivalize(np.asarray(deciles_av), na, nc) for na, nc in zip(nas, ncs)]).transpose()
df_income = pd.DataFrame(data_income, index=deciles_av.index, columns=df_comp.columns)

data_income_edges = np.asarray([dequivalize(np.asarray(deciles_edges), na, nc) for na, nc in zip(nas, ncs)]).transpose()
df_income_edges = pd.DataFrame(data_income_edges, index=deciles_edges.index, columns=df_comp.columns)

# what is earnt in each decile
df_incomesum = df_comp * df_income
df_incomesum_below = df_incomesum.cumsum()
df_incomesum_below.loc[0] = 0
df_incomesum_below.sort_index(inplace=True)
incomesum_below = df_incomesum_below.sum(axis=1) # we don't care about household breakdown now

df_incomesum_above = df_incomesum.sort_index(ascending=False).cumsum()
df_incomesum_above.index = df_incomesum_above.index - 0.1
df_incomesum_above.loc[1] = 0
df_incomesum_above.sort_index(inplace=True)
df_incomesum_above.index = df_incomesum_below.index #to get rid of some floating point errors introduced by the -0.1
incomesum_above = df_incomesum_above.sum(axis=1) # we don't care about household breakdown now

#print(incomesum_above + incomesum_below)
print(incomesum_above.loc[0]/1e12)

#what would be earnt if all below/above had same earnings as that decile
df_comp_below = df_comp.cumsum()
df_comp_below.loc[0] = 0
df_comp_below.sort_index(inplace=True)

df_required_incomesum_below = df_comp_below * df_income_edges
required_incomesum_below = df_required_incomesum_below.sum(axis=1) # we don't care about household breakdown now

df_comp_above = df_comp.sort_index(ascending=False).cumsum()
df_comp_above.index = df_comp_above.index - 0.1
df_comp_above.loc[1] = 0
df_comp_above.sort_index(inplace=True)
df_comp_above.index = df_income_edges.index #to get rid of some floating point errors introduced by the -0.1

df_required_incomesum_above = df_comp_above * df_income_edges
required_incomesum_above = df_required_incomesum_above.sum(axis=1) # we don't care about household breakdown now

required_incomesum = required_incomesum_below + required_incomesum_above

#these should be the same
n_households_bycomp = df_comp.sum()
n_households_bycomp_mat = np.asarray([np.asarray(n_households_bycomp) for id in range(11)])
df_n_households_bycomp_mat = pd.DataFrame(n_households_bycomp_mat, index=deciles_edges.index, columns=df_comp.columns)
required_incomesum2 = df_income_edges * df_n_households_bycomp_mat
required_incomesum2 = required_incomesum2.sum(axis=1)
print('sanity check results:')
print(required_incomesum - required_incomesum2)

f_pcInd_to_required_incomesum = interp1d(np.asarray(required_incomesum2.index), np.asarray(required_incomesum2))

#compare earnt above/below and required above below
deficit_below = required_incomesum_below - incomesum_below
excess_above = incomesum_above - required_incomesum_above
f_pcInd_to_deficit_below = interp1d(np.asarray(deficit_below.index), np.asarray(deficit_below))
f_pcInd_to_excess_above = interp1d(np.asarray(excess_above.index), np.asarray(excess_above))

print([deficit_below.loc[0.6]/1e12, excess_above.loc[0.6]/1e12])


f_pcInd_to_deficit_below = interp1d(np.asarray(deficit_below.index), np.asarray(deficit_below))
f_pcInd_to_excess_above = interp1d(np.asarray(excess_above.index), np.asarray(excess_above))

#things to save:
save_dict = {'f_HEDI_to_pcInd': f_HEDI_to_pcInd,
            'f_pcInd_to_HEDI' : f_pcInd_to_HEDI,
            'l_deciles_av' : l_deciles_av,
            'tot_households': tot_households,
            'f_pcInd_to_pcHouse' : f_pcInd_to_pcHouse,
            'f_pcInd_to_pcHouse_byComp': f_pcInd_to_pcHouse_byComp,
            'd_household_comps_to_index': d_household_comps_to_index,
		'f_pcInd_to_required_incomesum': f_pcInd_to_required_incomesum,
            'f_pcInd_to_deficit_below' : f_pcInd_to_deficit_below,
            'f_pcInd_to_excess_above': f_pcInd_to_excess_above
            }

save_file = 'income_assessment.pickle'
with open(save_file, 'wb') as f:
    pickle.dump(save_dict, f)
