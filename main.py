from sklearn.svm import OneClassSVM
from sklearn.cross_validation import cross_val_predict
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    del clients_loyalty['customer_id']
    stock_quantity_columns = []
    for stock_id in metadata.stock_ids:
        stock_name = metadata.stock_ids[stock_id][0]
        if stock_name == "None":
            continue
        add_N = clients_df[clients_df['stock_id'] == stock_id].groupby('customer_id')['quantity'].sum()
        if len(add_N) == 0:
            continue
        col_name = stock_name + '_quantity'
        stock_quantity_columns.append(col_name)
        add_N_df = add_N.to_frame(col_name)
        clients_loyalty = pd.merge(clients_loyalty, add_N_df, how='left', left_index=True, right_index=True)
        clients_loyalty[col_name] = np.nan_to_num(clients_loyalty[col_name])

        add_N2 = add_N * add_N
        add_N2_df = add_N2.to_frame('loyalty')
        clients_loyalty = pd.merge(clients_loyalty, add_N2_df, how='left', left_index=True, right_index=True)
        clients_loyalty['loyalty'] = clients_loyalty['loyalty_x'] + np.nan_to_num(clients_loyalty['loyalty_y'])
        del clients_loyalty['loyalty_x']
        del clients_loyalty['loyalty_y']

    clients_loyalty['loyalty'] **= 0.5

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
    chosen_columns = ['customer_id', 'loyalty', 'Products number', 'USD_total',
                      'Sale part'] + old_columns  # + stock_quantity_columns
    clients_df = clients_df[chosen_columns]
    print(clients_df.corr())
    # del clients_df['Products number']

    clients_df = clients_df.groupby('customer_id').mean().reset_index()
    print(clients_df)
    drop_index = clients_df['customer_id'].apply(
        lambda x: x not in [10402, 10518, 3513, 3795, 5632, 6372, 8325, 1031, 3384,
                            5586, 6308, 8249, 10100, 2857, 5555, 6265, 8183, 10009, 2850, 907, 1006, 37, 904])
    clients_df = clients_df.drop(drop_index)

    columns_without_id = list(clients_df.columns)
    columns_without_id.remove('customer_id')
    clients = clients_df[columns_without_id]

    n_clusters = 5
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # clustering_model = ClusterizationModel(model="agglomerative").fit(clients)
    # clustering_model.draw_clusters(method="", axis=ax1, show=False)
    # clustering_model = ClusterizationModel(model="dbscan", eps=5e4, min_samples=1).fit(clients)
    # clustering_model.draw_clusters(method="", axis=ax2, show=False)
    clustering_model = ClusterizationModel(n_clusters=n_clusters, model="KMeans").fit(clients)
    # clustering_model.draw_clusters(method="", show=True)
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
cluster_reseller_model.draw_clusters(method="", show=True)
print(cluster_reseller_model.get_mean_values())

print(resellers_grouped_clean_df.corr())
#
# x = resellers_grouped_df[['reseller_volume', 'amount_in_usd_x', 'amount_in_usd_y']]  # 'reseller_volume',
# y = resellers_grouped_df['reseller_discount']
#
# model = ensemble.RandomForestClassifier(n_jobs=4, n_estimators=25, random_state=0, max_depth=3,
#                                         max_leaf_nodes=15)
# predicted = cross_val_predict(model, x, y, cv=10, n_jobs=4)
# scores = cross_validation.cross_val_score(model, x, y, cv=10, n_jobs=4)
# mse = metrics.mean_squared_error(y, predicted)
# print(scores)
# print(mse)
# print(list(zip(predicted, y)))
# # plot_statistic(sorted(abs(predicted - y)), "residuals")

resellers_grouped_df['reseller_cluster'] = pd.Series(cluster_reseller_model.get_labels())
X = pd.merge(X, resellers_grouped_df[['reseller_id', 'reseller_cluster'] + clients_labels_full], how='left',
             on='reseller_id')
X = data_preprocess.preprocess_categorical(X, ['reseller_cluster'])

X = X.set_index(X['placed_date'])
X = X.sort_index()

# t = data_preprocess.normalize(X, metadata.normalize_columns)

stock_ids = list(filter(lambda stock_id: metadata.stock_ids[stock_id][0] != 'None', X["stock_id"].unique()))
stock_ids = list(filter(lambda stock_id: len(X[X["stock_id"] == stock_id]) >= 10, stock_ids))

short_names = {v: k for k, v in metadata.stock_short_name_ids.items()}
stock_names = list(map(lambda st_id: metadata.stock_ids[st_id][0], stock_ids))

stocks_mse = []
stocks_accuracy = []
stocks_acc_confidence_intervals = []

for stock_id in stock_ids:
    stock_rows = X[X["stock_id"] == stock_id].copy()

    print("stock_id: %s" % metadata.stock_ids[stock_id][0])

    idea_techs = []
    for column in stock_rows.columns:
        if column in metadata.tech_ids.values():
            if stock_rows.iloc[1][column] == 1:
                idea_techs.append(column)

    predictors = get_columns(X, ['stock_id', 'discount_desc', 'customer_status', 'license_type', 'reseller_discount',
                                 'reseller_volume', 'urban', 'population', 'HDI', 'avg_income', 'GDP', 'client_label',
                                 'reseller_cluster'])
    predictors.remove('stock_id')
    for tech in idea_techs:
        if tech in data.trends_manager.techs:
            tm = data.trends_manager.techs[tech]
            for series in tm.series:
                stock_rows = stock_rows.join(series, how='inner')

            predictors += list(series.name for series in tm.series if not pd.isnull(series)[0])

    x = stock_rows[predictors]

    from math import ceil

    mean_quantity = ceil(stock_rows['quantity'].mean())
    print("mean: %s" % mean_quantity)

    def quantity_mapper(quantity):
        if quantity <= 1.1:
            return 0
        if quantity <= 2 + 0.1:
            return 1
        return 2


    response = "QuantityClass"
    stock_rows[response] = stock_rows['quantity'].apply(quantity_mapper)

    y = stock_rows[response]

    def remove_if_constant(df, column):
        if df[column].isin(df[column].iloc[:1]).all():
            del df[column]

    for column in x.columns:
        remove_if_constant(x, column)

    if len(x) < 1:
        continue

    y = preproc.LabelEncoder().fit_transform(y)

    # idea_rows[predictors + [response]].to_csv("java_predict.csv")

    folds_cnt = 10
    if len(x) < 20:
        folds_cnt = 2
    model = ensemble.RandomForestClassifier(n_jobs=4, n_estimators=25, max_features=5, random_state=0, max_depth=3,
                                           max_leaf_nodes=15)
    predicted = cross_val_predict(model, x, y, cv=folds_cnt, n_jobs=4)

    scores = cross_validation.cross_val_score(model, x, y, cv=folds_cnt, n_jobs=4)
    stocks_accuracy.append(scores.mean())
    stocks_acc_confidence_intervals.append(scores.std() * 2)

    mse = metrics.mean_squared_error(y, predicted)
    stocks_mse.append(mse)

plot_statistic(stocks_mse, "MSE", stock_names)
plot_statistic(stocks_accuracy, "Accuracy", stock_names, stocks_acc_confidence_intervals)

# pd.tools.plotting.scatter_matrix(clients, alpha=0.2,
#                                  c='red', hist_kwds={'color': ['burlywood']})
# plt.show()
