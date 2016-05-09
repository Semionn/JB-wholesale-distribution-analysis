import metadata
from sklearn import preprocessing as preproc
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import gc
import os.path
import logging

TRENDS_PATH = "trends"
DATA_PATH = "data/"


class TechnologyManager:
    def __init__(self, tech_name):
        self.tech_name = tech_name
        self.series = []

    def add(self, series, source):
        series.name = self.tech_name + "(" + source + ")"
        self.series.append(series)

    def save(self, save_path):
        for num, series in enumerate(self.series):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, series.name + ".csv")
            series.to_csv(file_path)


class TrendsManager:
    def __init__(self, tech_ids):
        self.techs = {}
        self.tech_ids = tech_ids

    def add_tech(self, series, name=None, source=None, id=None):
        if pd.isnull(series.iloc[0]):
            return None
        if source is None:
            source = ""
        if id is not None:
            self.add_tech(self.tech_ids[id], series)
        elif name is not None:
            if name not in self.techs:
                self.techs[name] = TechnologyManager(name)
            self.techs[name].add(series, source)

    def save(self, save_path=TRENDS_PATH):
        for tech_manager in self.techs.values():
            tech_manager.save(save_path)

    def read(self, dir_path=TRENDS_PATH):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for file in files:
            name, ext = os.path.splitext(file)
            if ext != ".csv":
                continue
            series = pd.Series.from_csv(os.path.join(dir_path, file))
            if series is not None:
                source = name.split("(")[-1].rstrip(")")
                name = "(".join(name.split("(")[:-1])
                self.add_tech(series, name=name, source=source)


class DataManager:
    def __init__(self, main_data_frame, trends_manager=None):
        self.main_data_frame = main_data_frame
        self.trends_manager = trends_manager


def del_columns(data, column_names):
    for column_name in column_names:
        if column_name in data.columns:
            del data[column_name]


def preprocess_categorical(data, column_names):
    labeler = preproc.LabelEncoder()
    for column_name in column_names:
        try:
            data[column_name] = labeler.fit_transform(data[column_name])
            dummies_df = pd.get_dummies(data[column_name])
            dummies_df.columns = list(map(lambda x: column_name + "_" + str(x), dummies_df.columns))
            data = pd.merge(data, dummies_df, left_index=True, right_index=True)
        except:
            del data[column_name]
    return data


def normalize(dataframe, columns):
    for column in columns:
        dataframe[column] = preproc.scale(dataframe[column])
    return dataframe


def remove_nans(data):
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='any')
    for column_name in data.columns.values:
        try:
            data = data[np.isfinite(data[column_name])]
        except:
            logging.debug("column '%s' doesn't integral type " % column_name)
    return data


def get_resellers(file_path=DATA_PATH+"reseller_id_new.csv"):
    resellers = pd.read_csv(file_path)
    percent_columns = ["Percent_0", "Percent_5", "Percent_10", "Percent_15", "Percent_20"]

    def get_discount(x):
        if x.isnull().sum() == len(x):
            return 0
        return (resellers.columns.get_loc(x.argmax()) - 1) * 5

    resellers["discount"] = resellers[percent_columns].apply(get_discount, axis=1)
    resellers.rename(inplace=True, columns={'Reseller Code': 'reseller_id',
                                            'discount': 'reseller_discount',
                                            'Grand Total': 'reseller_volume'})
    resellers['reseller_id'].fillna(0, inplace=True)
    resellers['reseller_id'] = resellers['reseller_id'].astype(int)
    return resellers


def calc_trends(trends_file_path=DATA_PATH+"trend_dump_clear.txt"):
    gc.disable()
    f = open(trends_file_path)
    trends = defaultdict(lambda: defaultdict(lambda: []))
    date_line = []
    trends_ids = set()
    sources = set()
    for line in f:
        arr = line.split("\t")
        if len(arr) < 4:
            continue
        source = arr[0]
        trend_id = int(arr[1])
        date = pd.to_datetime(arr[2])
        date = pd.to_datetime(str(date.date()))
        value = float(arr[3])
        if date.year < 2009 or date.year > 2015 or trend_id not in metadata.used_tech:
            continue
        date_line.append(date)
        trends_ids |= {trend_id}
        sources |= {source}
        trends[source][trend_id].append([date, value])

    date_line = pd.date_range(min(date_line), max(date_line), freq='D')
    columns = ["Date"] + [x for x in map(lambda trend_id: metadata.tech_ids[trend_id], trends_ids)]
    trends_matr = defaultdict(lambda: defaultdict(lambda: []))
    for source in sources:
        for date in date_line:
            trends_matr[source][date] = [pd.to_datetime(date)] + [np.nan] * (len(columns) - 1)

    columns_dict = {}
    for i, trend_id in enumerate(trends_ids):
        columns_dict[trend_id] = i + 1

    for source in sources:
        for trend_id in trends[source]:
            for pair in trends[source][trend_id]:
                date = pair[0]
                value = pair[1]
                idx = columns_dict[trend_id]
                trends_matr[source][date][idx] = value
    trends = {}
    for source in sources:
        trends[source] = pd.DataFrame(list(trends_matr[source].values()), columns=columns)
        trends[source] = trends[source].set_index(pd.DatetimeIndex(trends[source]["Date"]))
        del trends[source]["Date"]
        trends[source] = trends[source].sort_index()

    gc.enable()

    trends_manager = TrendsManager(metadata.tech_ids)

    for source in sources:
        frame = trends[source]
        for column in trends[source].columns:
            column_mean = frame[column].ewm(span=12).mean()
            frame[column][np.abs(frame[column] - column_mean) > (0.05 * frame[column].std())] = np.nan
            frame[column].interpolate(method="values", inplace=True)
            frame[column][frame[column] == 0] = np.nan
            frame[column].interpolate(method="values", inplace=True)
        # fig = plt.figure()
        # fig.suptitle(source)
        # frame.plot()
        trends[source] = frame.ewm(span=12).mean()
        for column in trends[source].columns:
            trends_manager.add_tech(trends[source][column], name=column, source=source)
            # roll = frame.rolling(window=12)
            # frame.plot()
            # roll.mean().plot()
    # plt.show()

    return trends_manager


def save_data(dataframe, file_name=DATA_PATH+"preprocessed_data.csv"):
    dataframe.to_csv(file_name)


def calc_data(file_name=DATA_PATH+'resellers_data.csv', only_resellers=True):
    logging.info("start process file %s" % file_name)

    data = pd.read_csv(file_name, low_memory=False)
    logging.info("reading file %s complete" % file_name)

    if only_resellers:
        data = data[data['reseller_id'] != 0]
        logging.info("direct sells filtered")

    data['placed_date'] = pd.to_datetime(data['placed_date'])

    logging.info("date index is set")
    stock_shorts = metadata.stock_short_name_ids
    stock_ids = metadata.stock_ids

    def stocks_mapper(stock_name):
        return next(x for x in stock_shorts.items() if stock_name.startswith(x[0]))[1]

    def main_tech_mapper(row):
        return stock_ids[row["stock_id"]][1]

    data["stock_id"] = data["stock_id"].apply(stocks_mapper)
    logging.info("stock_id evaluated")
    data["main_tech_id"] = data.apply(main_tech_mapper, axis=1)
    logging.info("main_tech_id evaluated")

    v = DictVectorizer()
    techs_df = v.fit_transform(metadata.stock_ids_table)
    techs_df = pd.DataFrame(techs_df.todense(), columns=v.get_feature_names())
    data = pd.merge(data, techs_df, on='stock_id')
    logging.info("additional techs evaluated")

    v2 = DictVectorizer()
    countries = v2.fit_transform(metadata.countries_data)
    countries = pd.DataFrame(countries.todense(), columns=v2.get_feature_names())
    countries["iso"] = list(map(lambda d: d["iso"], metadata.countries_data))
    logging.info("countries data prepared")
    data = pd.merge(data, countries, on='iso')
    logging.info("countries data merged")

    data['discount_desc'] = data['discount_desc'].notnull()
    data = preprocess_categorical(data, metadata.preprocess_columns)
    logging.info("categorical data evaluated")
    del_columns(data, metadata.del_column_names)
    logging.info("worse columns removed")

    resellers_df = get_resellers()[['reseller_id', 'reseller_volume', 'reseller_discount']]
    logging.info("resellers data prepared")

    data['reseller_id'].fillna(0, inplace=True)
    data['reseller_id'] = data['reseller_id'].astype(int)
    data = pd.merge(data, resellers_df, on='reseller_id')

    logging.info("resellers data merged")

    data = data.set_index(pd.DatetimeIndex(data['placed_date']))
    data = data.sort_index()

    data = remove_nans(data)
    logging.info("nans removed")
    return data


def get_trends(calc=False, trends_file_path=DATA_PATH+"trend_dump_clear.txt", path=TRENDS_PATH):
    if calc:
        trends_manager = calc_trends(trends_file_path)
        trends_manager.save(TRENDS_PATH)
        return trends_manager
    trends_manager = TrendsManager(metadata.tech_ids)
    trends_manager.read(path)
    return trends_manager


def get_data(calc=False, calc_trends=False, preproc_file_name=DATA_PATH+"preprocessed_data.csv",
             file_name=DATA_PATH+"purchases_org_resellers.csv"):
    if calc:
        df = calc_data(file_name)
        save_data(df)
    else:
        df = pd.read_csv(preproc_file_name)
        df.rename({"Unnamed: 0": "placed_date"}, inplace=True)
        del df["Unnamed: 0"]
        df = df.set_index(pd.DatetimeIndex(df["placed_date"]))
    trends_manager = get_trends(calc_trends)
    return DataManager(df, trends_manager)
