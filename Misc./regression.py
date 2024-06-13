import pandas as pd

run_nums = list(range(1, 4))
participant_ids = [407, 408] + list(range(2101, 2112))

def get_file_name(pid, run_num):
    return f"./{pid}/results/{pid}_{run_num}_rmse_results.csv"
results = {pid: {run_num: pd.read_csv(get_file_name(pid, run_num)) for run_num in run_nums} for pid in participant_ids}

results_df = pd.DataFrame(columns=['pid', 'run_num', 'distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse'])

for pid in participant_ids:
    for run_num in run_nums:
        distance_rmse = results[pid][run_num]['distance_rmse'].mean()
        speed_difference_rmse = results[pid][run_num]['speed_difference_rmse'].mean()
        direction_difference_rmse = results[pid][run_num]['direction_difference_rmse'].mean()
        results_df = results_df.append({'pid': pid, 'run_num': run_num, 'distance_rmse': distance_rmse, 'speed_difference_rmse': speed_difference_rmse, 'direction_difference_rmse': direction_difference_rmse}, ignore_index=True)

results_df = results_df.astype({'pid': 'int32', 'run_num': 'int32', 'distance_rmse': 'float32', 'speed_difference_rmse': 'float32', 'direction_difference_rmse': 'float32'})
# results_df.to_csv('results_all.csv', index=False)

qualtrics_df = pd.read_csv('qualtrics_dataa.csv')
qualtrics_df = qualtrics_df.rename(columns={'P_NUM': 'pid'})
to_drop =['Question ID', 'Age', 'Sex', 'GSQ1_1', 'HSF1_1', 'HSF1_2',
       'HSF1_3', 'HSF2_1', 'HSF3_1', 'HSF3_2', 'HSF4_1', 'HSF6_1', 'HSF6_2',
       'S1_1', 'S2_1', 'S3_1', 'S4_1', 'S5_1', 'mental-model_open', 'SART1_1',
       'SART2_1', 'SART3_1', 'SART4_1', 'SART5_1', 'SART6_1', 'SART7_1',
       'SART8_1', 'SART9_1', 'CQQ1_1', 'CQQ2_1', 'SART10_1', 'CQQ_open',
       'NATX1-3_1', 'NATX1-3_2', 'NATX1-3_4', 'NATX4-6_5', 'NATX4-6_6',
       'NATX4-6_7']
qualtrics_df = qualtrics_df.drop(columns=to_drop).iloc[1:, :]
qualtrics_df = qualtrics_df.astype({'pid': 'int32'})

# add columns 'A', 'B', 'C' to results_df
results_df['A'] = None
results_df['B'] = None
results_df['C'] = None


for pid in participant_ids:
    for run_num in run_nums:
        # print(results_df.loc[(results_df['pid'] == pid) & (results_df['run_num'] == run_num), ['A', 'B', 'C']])
        results_df.loc[(results_df['pid'] == pid) & (results_df['run_num'] == run_num), ['A', 'B', 'C']] = qualtrics_df.loc[qualtrics_df['pid'] == pid, [f'{run_num}A_1', f'{run_num}B_1', f'{run_num}C_1']].to_numpy()
        # print(results_df.loc[(results_df['pid'] == pid) & (results_df['run_num'] == run_num), ['A', 'B', 'C']], qualtrics_df.loc[qualtrics_df['pid'] == pid, [f'{run_num}A_1', f'{run_num}B_1', f'{run_num}C_1']])

# results_df = results_df.astype({'A': 'int32', 'B': 'int32', 'C': 'int32'})

# print(qualtrics_df.head())
# print(results_df.head())

results_df.to_csv('results_all.csv', index=False)
qualtrics_df.to_csv('qualtrics_data.csv', index=False)

from sklearn import linear_model

X = results_df[['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse']].to_numpy()
for q in ['A', 'B', 'C']:
    y = results_df[q].to_numpy()
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    print(f'Question {q}:')
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)
    print('R^2: ', regr.score(X, y))
    print('')

# correlation heatmap for all questions
import seaborn as sns
import matplotlib.pyplot as plt

corr = results_df.astype('float32').corr()
print(corr.loc[['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse'],['A', 'B', 'C']])
plt.title('Correlation Heatmap')
ax = sns.heatmap(corr.loc[['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse'],['A', 'B', 'C']], annot=True)
ax.figure.tight_layout()
plt.savefig('correlation_heatmap.png')

