import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

run_nums = list(range(1, 4))
participant_ids = [407, 408] + list(range(2101, 2112))
graph_names = ['paths', 'direction_difference', 'distance', 'pace_asymmetry']
rmse_names = [' ', 'direction_difference_rmse', 'distance_rmse', 'speed_difference_rmse']

def get_graph_names(pid, run_num):
    return [f"./{pid}/results/{pid}_{run_num}_{graph}.png" for graph in graph_names]

def make_image(res_table, r):
    table = plt.table(cellText=res_table.T, rowLabels=['  A  ', '  B  ', '  C  '], colLabels=['Rating'], cellLoc = 'center', rowLoc = 'center',
          loc='center')
    # return an image of the table
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'temp_{r}.png')
    plt.close()

results_all = pd.read_csv('./results_all.csv')


for pid in participant_ids:
    for i, run_num in enumerate(run_nums):
        for j, (graph_name, rmse_name) in enumerate(zip(graph_names, rmse_names)):
            res_table = results_all.loc[(results_all['pid'] == pid) & (results_all['run_num'] == run_num), ['A', 'B', 'C']].to_numpy()
            make_image(res_table, i)
    fig = plt.figure(figsize=(35., 20.))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(3, 5),
                 axes_pad=0.5,
                 )
    for i, run_num in enumerate(run_nums):
        for j, (graph_name, rmse_name) in enumerate(zip(graph_names, rmse_names)):
            grid[i*5 + j].imshow(plt.imread(get_graph_names(pid, run_num)[j]))
            if j != 0:
                grid[i*5 + j].set_title(str(results_all.loc[(results_all['pid'] == pid) & (results_all['run_num'] == run_num), rmse_name].to_numpy()[0]), fontsize=25)
            grid[i*5 + j].axis('off')
        res_table = results_all.loc[(results_all['pid'] == pid) & (results_all['run_num'] == run_num), ['A', 'B', 'C']].to_numpy()
        grid[i*5 + 4].imshow(plt.imread(f'temp_{i}.png'))
        grid[i*5 + 4].axis('off')
    plt.suptitle(f'Participant {pid}', fontsize=40)
    plt.tight_layout()
    plt.savefig(f'./{pid}/results/combined.png')