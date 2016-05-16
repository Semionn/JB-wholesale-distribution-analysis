from urllib import request
import zipfile
import pandas as pd
import metadata

URBAN_PERCENT_CSV = "http://api.worldbank.org/v2/en/indicator/sp.urb.totl.in.zs?downloadformat=csv"
GDP_DOLLARS_SCV = "http://api.worldbank.org/v2/en/indicator/ny.gdp.mktp.pp.cd?downloadformat=csv"
POPULATION_CSV = "http://api.worldbank.org/v2/en/indicator/sp.pop.totl?downloadformat=csv"


def load_file(addr, file_name="data/temp.zip"):
    request.urlretrieve(addr, file_name)
    return file_name


def extract_zip(path, condition):
    fh = open(path, 'rb')
    z = zipfile.ZipFile(fh)
    name = next(filter(lambda x: condition(x), z.namelist()))
    outpath = "data/"
    z.extract(name, outpath)
    fh.close()
    return outpath + name


def print_full_df(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')


def add_feature(df, df2, feature):
    df = pd.merge(df, df2, on='name')
    df.loc[df[feature+"_x"] != 0, feature+'_y'] = 0
    df[feature] = df[feature+"_x"] + df[feature+"_y"]
    del df[feature+"_x"]
    del df[feature+"_y"]
    return df

countries = pd.DataFrame([], columns=["iso", "name", "region", "urban", "population", "HDI", "avg_income", "GDP"])
for country in metadata.countries_data:
    countries = countries.append(country, ignore_index=True)


# calc gdp data
file_gdp_zip = load_file(GDP_DOLLARS_SCV)
file_gdp_csv = extract_zip(file_gdp_zip, lambda x: x.startswith("API"))
df = pd.read_csv(file_gdp_csv, skiprows=4)
df = df[['Country Name', '2014']]
feature = 'GDP'
df.columns = ['name', feature]
df = df.dropna(axis=0, how='any')
df['name'] = df['name'].apply(lambda s: s.upper())
df[feature] = df[feature].apply(lambda x: int(x/1e6))
countries = add_feature(countries, df, feature)


# calc urban data
file_urban_zip = load_file(URBAN_PERCENT_CSV)
file_urban_zip = extract_zip(file_urban_zip, lambda x: x.startswith("API"))
df = pd.read_csv(file_urban_zip, skiprows=4)
df = df[['Country Name', '2014']]
feature = 'urban'
df.columns = ['name', feature]
df = df.dropna(axis=0, how='any')
df['name'] = df['name'].apply(lambda s: s.upper())
df[feature] = df[feature].apply(lambda x: int(x)/100)
countries = add_feature(countries, df, feature)


# calc population data
file_urban_zip = load_file(POPULATION_CSV)
file_urban_zip = extract_zip(file_urban_zip, lambda x: x.startswith("API"))
df = pd.read_csv(file_urban_zip, skiprows=4)
df = df[['Country Name', '2014']]
feature = 'population'
df.columns = ['name', feature]
df = df.dropna(axis=0, how='any')
df['name'] = df['name'].apply(lambda s: s.upper())
df[feature] = df[feature].apply(lambda x: int(x))
countries = add_feature(countries, df, feature)


# calc salaries data
df = pd.read_csv("data/annual_salaries.csv")
df = df[['Country Name', '2012']]
feature = 'avg_income'
df.columns = ['name', feature]
df = df.dropna(axis=0, how='any')
df['name'] = df['name'].apply(lambda s: s.upper())
df[feature] = df[feature].apply(lambda x: int(x))
countries = add_feature(countries, df, feature)


# calc HDI data
df = pd.read_csv("data/hdi_index.csv")
df = df[['Country Name', '2014']]
feature = 'HDI'
df.columns = ['name', feature]
df = df.dropna(axis=0, how='any')
df['name'] = df['name'].apply(lambda s: s.upper())
countries = add_feature(countries, df, feature)
countries.to_csv("data/countries_stats.csv", index=False)
print_full_df(countries)