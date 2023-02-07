import pandas as pd
import numpy as np

INDIR = r'./'
filename = INDIR + 'equivalized_disposable_income_deciles_timeseries.xlsx'


# decile points, in terms of equivalized household income (to 2 adult household)
# by individual (i.e. each decile has equal number of people, not equal number of households)
sheetname = 'data'
df = pd.read_excel(filename, sheetname, header=0)
df.drop(columns='Year', inplace=True)
df = df.set_index(np.arange(1977, 2021))

df_norm = df.copy()
for col in df:
    df_norm[col] = df[col]/df.loc[1977, col]

#better to do CAGR...
df['year_diff'] = 2020 - df.index 
df.sort_index(ascending=False, inplace=True)

df_cagr = df.copy()

for col in df:
    df_cagr[col] = np.power(df.loc[2020, col]/df[col], 1/df['year_diff'])

def write_xlsx(filename, df):
    xlsx = pd.ExcelWriter(filename)
    df.to_excel(xlsx, 'data')    
    xlsx.close()

write_xlsx('CAGR_decile_points.xlsx', df_cagr)
write_xlsx('norm_decile_points.xlsx', df_norm)

# Now for UK economy
df = pd.read_csv('uk_gdp_growth.csv', index_col=0, skiprows=8, names=['Year', 'PCGrowth'])
data = df.loc[1977:2021, 'PCGrowth'].to_numpy()
data = 1 + data/100
uk_gdp_cagr = np.power(np.prod(data), 1/data.size)
print(uk_gdp_cagr)