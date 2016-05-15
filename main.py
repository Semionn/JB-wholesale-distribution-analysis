from sklearn.linear_model.base import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing.data import StandardScaler, PolynomialFeatures
from sklearn.svm import OneClassSVM
from sklearn.cross_validation import cross_val_predict
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm.classes import SVR

import data_preprocess
import metadata
import logging
from clusterization_model import ClusterizationModel
from sklearn import preprocessing as preproc

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

# reload(logging)
logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def remove_columns(df, columns):
    new_columns = list(filter(lambda x: x not in columns, df.columns))
    return df[new_columns]


def get_columns(df, columns, get_all_match=True):
    new_columns = []
    for column in columns:
        all_match = list(filter(lambda x: column in x, df.columns))
        if get_all_match or column not in df.columns:
            new_columns += all_match
        else:
            new_columns.append(column)
    return new_columns


def plot_statistic(statistic, title, names=None, conf_interv=None):
    fig, ax = plt.subplots()
    index = np.arange(len(statistic))
    plt.bar(index, statistic, yerr=conf_interv, color='b', alpha=0.7)
    if names is not None:
        plt.xticks(index + 0.4, names)
    # plt.xlabel("Stocks")
    plt.ylabel(title)
    plt.show()


data = data_preprocess.get_data()

X = data.main_data_frame


def remove_outliers(data):
    clf = OneClassSVM(nu=0.2, kernel="rbf", gamma=0.00001)
    clf.fit(data)
    logging.info("%s outliers removed from %s elements" % ((clf.predict(data) == -1).sum(), len(data)))
    return data[clf.predict(data) == 1]


def clients_clusterization(X):
    clients_df = X[get_columns(X, ['stock_id', 'customer_id', 'quantity', 'customer_status', 'license_type', 'iso',
                                   'amount_in_usd', 'GDP', 'HDI', 'population', 'urban', 'avg_income',
                                   'discount_perc'])]
    clients_df.reset_index(inplace=True)

    clients_loyalty = clients_df.groupby('customer_id')[['customer_id']].mean()
    clients_loyalty['loyalty'] = 0
    clients_loyalty['Renew loyalty'] = 0
    del clients_loyalty['customer_id']
    stock_quantity_columns = []
    for stock_id in metadata.stock_ids:
        stock_name = metadata.stock_ids[stock_id][0]
        if stock_name == "None":
            continue
        add_N = clients_df[clients_df['stock_id'] == stock_id].groupby('customer_id')['quantity'].sum()
        add_Renew = clients_df[(clients_df['stock_id'] == stock_id) &
                               (clients_df['license_type'] == 2)].groupby('customer_id')['quantity'].count()
        if len(add_N) == 0:
            continue
        col_name = stock_name + '_quantity'
        stock_quantity_columns.append(col_name)
        add_N_df = add_N.to_frame(col_name)
        clients_loyalty = pd.merge(clients_loyalty, add_N_df, how='left', left_index=True, right_index=True)
        clients_loyalty[col_name] = np.nan_to_num(clients_loyalty[col_name])

        add_N2 = add_N * add_N
        add_N2_df = add_N2.to_frame('loyalty')
        add_Renew2 = add_Renew * add_Renew
        add_Renew2_df = add_Renew2.to_frame('Renew loyalty')
        clients_loyalty = pd.merge(clients_loyalty, add_N2_df, how='left', left_index=True, right_index=True)
        clients_loyalty['loyalty'] = clients_loyalty['loyalty_x'] + np.nan_to_num(clients_loyalty['loyalty_y'])
        clients_loyalty = pd.merge(clients_loyalty, add_Renew2_df, how='left', left_index=True, right_index=True)
        clients_loyalty['Renew loyalty'] = clients_loyalty['Renew loyalty_x'] + np.nan_to_num(
            clients_loyalty['Renew loyalty_y'])
        del clients_loyalty['loyalty_x']
        del clients_loyalty['loyalty_y']
        del clients_loyalty['Renew loyalty_x']
        del clients_loyalty['Renew loyalty_y']

    clients_loyalty['loyalty'] **= 0.5
    clients_loyalty['Renew loyalty'] **= 0.5

    clients_df = pd.merge(clients_df, clients_loyalty, how='left', left_on='customer_id', right_index=True)

    def calc_feature(df_from, df_to, base_param_name, new_param_name, group_id='customer_id'):
        df_from = df_from.groupby(group_id).sum().reset_index()
        df_from[new_param_name] = df_from[base_param_name]
        result = pd.merge(df_to, df_from[[group_id, new_param_name]], how='left', on=group_id)
        result[new_param_name] = np.nan_to_num(result[new_param_name])
        return result

    clients_new_pursh = clients_df[['customer_id', 'quantity', 'license_type']].loc[clients_df['license_type'] == 0]
    clients_df = calc_feature(clients_new_pursh, clients_df, 'quantity', 'Products number')

    clients_df = calc_feature(clients_df[['customer_id', 'amount_in_usd']], clients_df, 'amount_in_usd', 'USD_total')

    clients_sale_filtered = clients_df[['customer_id', 'amount_in_usd']][clients_df['discount_perc'] > 10]
    clients_df = calc_feature(clients_sale_filtered, clients_df, 'amount_in_usd', 'Dicsount_amount_USD')

    clients_df['Sale part'] = clients_df['Dicsount_amount_USD'] / clients_df['USD_total']

    old_columns = ['GDP', 'HDI', 'population', 'urban', 'avg_income']
    chosen_columns = ['customer_id', 'loyalty', 'Products number', 'USD_total', 'Renew loyalty',
                      'Sale part'] + old_columns  # + stock_quantity_columns
    clients_df = clients_df[chosen_columns]
    print(clients_df.corr())

    clients_df = clients_df.groupby('customer_id').mean().reset_index()
    print(clients_df)
    # drop few outliers, which have been found by hierarchy clusterization
    drop_index = clients_df['customer_id'].apply(
        lambda x: x not in [10402, 10518, 3513, 3795, 5632, 6372, 8325, 1031, 3384,
                            5586, 6308, 8249, 10100, 2857, 5555, 6265, 8183, 10009, 2850, 907, 1006, 37, 904])
    clients_df = clients_df.drop(drop_index)

    columns_without_id = list(clients_df.columns)
    columns_without_id.remove('customer_id')
    clients = clients_df[columns_without_id]

    n_clusters = 3
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # clustering_model = ClusterizationModel(model="agglomerative").fit(clients)
    # clustering_model.draw_clusters(method="", axis=ax1, show=False)
    # clustering_model = ClusterizationModel(model="dbscan", eps=5e4, min_samples=1).fit(clients)
    # clustering_model.draw_clusters(method="", axis=ax2, show=False)
    clustering_model = ClusterizationModel(n_clusters=n_clusters, model="KMeans").fit(clients)
    clustering_model.draw_clusters(method="", show=True)
    # clustering_model = ClusterizationModel(model="hierarchy").fit(clients)
    # clustering_model.draw_clusters(method="dendrogram", show=True)

    mean_values = clustering_model.get_mean_values()
    mean_values = mean_values.sort_values(by='loyalty', ascending=[1])

    labels = clustering_model.get_labels()
    clients_df['label'] = pd.Series(labels)
    return mean_values, clients_df


mean_values, clients_df = clients_clusterization(X)
print(mean_values)

resellers_df = X[['reseller_id', 'reseller_volume', 'reseller_discount', 'amount_in_usd', 'customer_id']]
clusters_labels = list(clients_df['label'].unique())
clients_labels = clients_df[['customer_id', 'label']]
clients_labels.columns = ['customer_id', 'client_label']
resellers_grouped_df_mean = resellers_df.groupby('reseller_id').mean().reset_index()
resellers_df = pd.merge(resellers_df, clients_labels, on='customer_id')
resellers_df = data_preprocess.preprocess_categorical(resellers_df, ['client_label'])

clients_labels_full = list(filter(lambda x: 'client_label' in x, resellers_df.columns))
resellers_grouped_df_sum = resellers_df[['reseller_id', 'amount_in_usd'] + clients_labels_full].groupby(
    'reseller_id').sum().reset_index()
resellers_grouped_df = pd.merge(resellers_grouped_df_mean, resellers_grouped_df_sum, on='reseller_id')

resellers_grouped_clean_df = remove_columns(resellers_grouped_df, ['reseller_id', 'customer_id'])
n_clusters = 5
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# cluster_reseller_model = ClusterizationModel(model="agglomerative").fit(resellers_grouped_clean_df)
# cluster_reseller_model.draw_clusters(method="", axis=ax1, show=False)
# cluster_reseller_model = ClusterizationModel(model="dbscan", eps=5e4, min_samples=1).fit(resellers_grouped_clean_df)
# cluster_reseller_model.draw_clusters(method="", axis=ax2, show=False)
cluster_reseller_model = ClusterizationModel(n_clusters=n_clusters, model="KMeans").fit(resellers_grouped_clean_df)
# cluster_reseller_model.draw_clusters(method="", show=True)
# cluster_reseller_model = ClusterizationModel(model="hierarchy").fit(resellers_grouped_clean_df)
# cluster_reseller_model.draw_clusters(method="dendrogram", show=True)
print(cluster_reseller_model.get_mean_values())

print(resellers_grouped_clean_df.corr())

# plot resellers volume, grouped by discount
discount_groups = resellers_grouped_df[['reseller_volume', 'reseller_discount']].groupby('reseller_discount').agg(
    ['mean', 'count'])
discount_groups['reseller_volume'].plot(kind='bar', subplots=True)
plt.ylabel('reseller_volume')
plt.show()

resellers_grouped_df['reseller_cluster'] = pd.Series(cluster_reseller_model.get_labels())
X = pd.merge(X, resellers_grouped_df[['reseller_id', 'reseller_cluster'] + clients_labels_full], how='left',
             on='reseller_id')

import statsmodels.api as sm
from scipy import stats
reseller_clusters = list(X['reseller_cluster'].unique())
for cluster in reseller_clusters:
    resellers_df = X[['reseller_cluster', 'amount_in_usd', 'placed_date']][X['reseller_cluster'] == cluster]
    resellers_df['placed_date'] = pd.to_datetime(resellers_df['placed_date'])
    resellers_df = resellers_df[(resellers_df['placed_date'] >= '2012-01-01') & (resellers_df['placed_date'] < '2016-01-01')]
    resellers_df = resellers_df.set_index(resellers_df['placed_date'])
    resellers_df = resellers_df.sort_index()
    resellers_df = resellers_df[['amount_in_usd']]
    resellers_df = resellers_df.groupby(pd.TimeGrouper(freq='W')).sum()
    resellers_df.loc[pd.to_datetime('2014-12-31')] = float(np.nan)
    resellers_df.loc[pd.to_datetime('2015-12-31')] = float(np.nan)
    resellers_df['amount_in_usd'].interpolate(method="values", inplace=True)
    time_series_df = pd.DataFrame(resellers_df[['amount_in_usd']])
    time_series_df.plot(figsize=(15,5))
    plt.show()
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(resellers_df.values.squeeze(), lags=48, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(resellers_df, lags=48, ax=ax2)
    # arma_mod30 = sm.tsa.ARMA(resellers_df, (12, 3)).fit()
    # print(arma_mod30.params)
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax = resellers_df.ix['2012':].plot(ax=ax)
    # fig = arma_mod30.plot_predict('2014', '2015', dynamic=True, ax=ax, plot_insample=False)