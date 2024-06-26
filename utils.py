import pandas as pd
import numpy as np
from geopy.distance import geodesic
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
from scipy import signal
import dtw

DEBUG = None

collectdict = {}

############################################################################################################
""" Functions """

def clean_DS_store_files():
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name == ".DS_Store":
                os.remove(os.path.join(root, name))
        for name in dirs:
            if name == ".DS_Store":
                os.rmdir(os.path.join(root, name))

def line_plot(df, x, y, title, xlabel, ylabel, save_path):
    if DEBUG:
        return
    plt.clf()
    if type(y) != str:
        colors = ["coral", "blue"]
        labels = ["participant", "robot"]
        for i, col in enumerate(y):
            sns.lineplot(x=x, y=col, color=colors[i], data=df, label=labels[i])
        plt.legend()
    else:
        sns.lineplot(x=x, y=y, color="green", data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)

def get_file_paths(parent_dir):
    files = sorted(os.listdir(parent_dir))
    robot_indexes = [i for i, s in enumerate(files) if "robot" in s]
    robot_files = sorted([files[i] for i in robot_indexes])
    participant_indexes = [i for i, s in enumerate(files) if "robot" not in s]
    participant_files = sorted([files[i] for i in participant_indexes])
    return participant_files, robot_files

def create_folder(path, instead=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif instead:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        
def create_base_folders(participant_id, instead=True):
    results_parent_folder = f"./results/"
    turns_parent_folder = f"./turns/"
    create_folder(results_parent_folder, instead=instead)
    create_folder(turns_parent_folder, instead=instead)
    results_folder = results_parent_folder + f"{participant_id}/"
    turns_folder = turns_parent_folder + f"{participant_id}/"
    create_folder(results_folder, instead=instead)
    create_folder(turns_folder, instead=instead)
    turns_folder_person = turns_folder + "person/"
    turns_folder_robot = turns_folder + "robot/"
    create_folder(turns_folder_person, instead=instead)
    create_folder(turns_folder_robot, instead=instead)
    return results_folder, turns_folder

def read_csv_files(parent_dir, path_num, verbose=True):
    participant_files, robot_files = get_file_paths(parent_dir)
    person_csv_file = parent_dir + f"{participant_files[path_num]}"
    robot_csv_file = parent_dir + f"{robot_files[path_num]}"
    if verbose:
        print(person_csv_file, robot_csv_file)
    df_person = pd.read_csv(person_csv_file)
    df_robot = pd.read_csv(robot_csv_file)
    return df_person, df_robot

def preprocess_join(df_person, df_robot):

    df_person["date"] = pd.to_datetime(df_person["date"]).dt.tz_localize(None)
    df_robot["date"] = pd.to_datetime(df_robot["date"]).dt.tz_localize(None)

    # Find the minimum date common to both CSV files
    min_date = max(df_person["date"].min(), df_robot["date"].min())

    df_person = df_person[df_person["date"] >= min_date]
    df_robot = df_robot[df_robot["date"] >= min_date]

    # Extract the required columns
    df_person_data = df_person[["date", "GPS (Lat.) [deg]", "GPS (Long.) [deg]", "GPS (2D) [m/s]"]]
    df_robot_data = df_robot[["date", "GPS (Lat.) [deg]", "GPS (Long.) [deg]", "GPS (2D) [m/s]"]]

    df_person_data.columns = ["date", "GPS1 (Lat.) [deg]", "GPS1 (Long.) [deg]", "GPS1 (2D) [m/s]"]
    df_robot_data.columns = ["date", "GPS2 (Lat.) [deg]", "GPS2 (Long.) [deg]", "GPS2 (2D) [m/s]"]

    # Merge the dataframes based on the "date" column
    df = pd.merge(df_person_data, df_robot_data, on="date")
    
    # find first index where robot moves
    threshold = 0.1
    first_move_index = 0
    for j in range(len(df["GPS2 (2D) [m/s]"])):
        if df.loc[j,"GPS2 (2D) [m/s]"] > threshold:
            first_move_index = j
            break

    # remove first_move_index rows from df
    df = df.iloc[first_move_index:]
    df.reset_index(drop=True, inplace=True)

    # find the minimum length of the two dataframes
    min_length = min(len(df[col]) for col in df.columns)
    df = df.iloc[:min_length]
    df.reset_index(drop=True, inplace=True)
    
    return df

def add_cols(df):
    df["date"] = pd.to_datetime(df["date"])
    first_date = df.iloc[0]["date"]
    df["milliseconds"] = (df["date"] - first_date).dt.total_seconds() * 1000
    df["seconds"] = (df["date"] - first_date).dt.total_seconds()

    df["diff_milliseconds"] = df["milliseconds"].diff()
    df["diff_seconds"] = df["seconds"].diff()

    df["distance"] = df.apply(lambda row: geodesic((row["GPS1 (Lat.) [deg]"], row["GPS1 (Long.) [deg]"]), (row["GPS2 (Lat.) [deg]"], row["GPS2 (Long.) [deg]"])).meters, axis=1)
    # df["distance_target"] = df["distance"].apply(lambda x: x if x > 1 else 0)

    df["speed_difference"] = abs(df["GPS1 (2D) [m/s]"] - df["GPS2 (2D) [m/s]"])

    df["pace_difference"] = abs(df["GPS1 (2D) [m/s]"] - df["GPS2 (2D) [m/s]"]) / (df["GPS1 (2D) [m/s]"] + df["GPS2 (2D) [m/s]"])

    df["walking_direction"] = np.arctan2(df["GPS1 (Lat.) [deg]"].diff(), df["GPS1 (Long.) [deg]"].diff())
    df["walking_direction_robot"] = np.arctan2(df["GPS2 (Lat.) [deg]"].diff(), df["GPS2 (Long.) [deg]"].diff())
    
    df["direction_difference"] = abs(df["walking_direction"] - df["walking_direction_robot"])
    df["direction_difference"] = np.where(df["direction_difference"] > 180, df["direction_difference"] - 180, df["direction_difference"])
    
    return df

def basic_plots(df, pre):
    if DEBUG:
        return
    # plotting distance over time
    line_plot(df, "seconds", "distance", "Distance Between Robot and Participant", "Time (s)", "Distance (m)", f"{pre}distance.png")

    # plotting speeds over time
    line_plot(df, "seconds", ["GPS1 (2D) [m/s]", "GPS2 (2D) [m/s]"], "Speed Plot", "Time (s)", "Speed (m/s)", f"{pre}speeds.png")

    # plotting speed difference over time
    line_plot(df, "seconds", "speed_difference", "Speed Difference", "Time (s)", "Speed Difference (m/s)", f"{pre}speed_difference.png")

    # plotting pace difference over time
    line_plot(df, "seconds", "pace_difference", "Pace Asymmetry", "Time (s)", "Pace Asymmetry", f"{pre}pace_asymmetry.png")

    # plot walking directions of participant and robot
    line_plot(df, "seconds", ["walking_direction", "walking_direction_robot"], "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction_original.png")    

    # plot direction difference over time
    line_plot(df, "seconds", "direction_difference", "Angle of Difference in Walking Direction", "Time (s)", "Direction Difference (deg)", f"{pre}direction_difference.png")

def basic_rmse(df, pre, verbose=True):
    rmse_results = []
    # calculate total distance RMSE
    distance_rmse = math.sqrt(np.sum((df["distance_target"]**2)*df["diff_milliseconds"])/df["diff_milliseconds"].sum())
    if verbose:
        print(f"distance target RMSE is {distance_rmse}")
    rmse_results.append(distance_rmse)

    # printing speed difference RMSE
    speed_difference_rmse = math.sqrt(np.sum((df["speed_difference"]**2)*df["diff_milliseconds"])/df["diff_milliseconds"].sum())
    if verbose:
        print(f"speed difference RMSE is {speed_difference_rmse}")
    rmse_results.append(speed_difference_rmse)

    # printing direction difference RMSE
    direction_difference_rmse = math.sqrt(np.sum((df["direction_difference"]**2)*df["diff_milliseconds"])/df["diff_milliseconds"].sum())
    if verbose:
        print(f"direction difference RMSE is {direction_difference_rmse}")
    rmse_results.append(direction_difference_rmse)

    # save RMSE results to csv
    cols = ["distance_rmse", "speed_difference_rmse", "direction_difference_rmse"]
    rmse_results_df = pd.DataFrame([rmse_results], columns=cols)
    rmse_results_df.to_csv(f"{pre}rmse_results.csv", index=False)
    
    return rmse_results

def fix_walking_direction(df):
    tmp_dir = df["walking_direction"].values
    for j in range(1, len(tmp_dir)):
        if tmp_dir[j-1] > 0 and tmp_dir[j] < 0 and tmp_dir[j-1] - tmp_dir[j] > np.pi:
            tmp_dir[j] += 2*np.pi
        elif tmp_dir[j-1] < 0 and tmp_dir[j] > 0 and tmp_dir[j] - tmp_dir[j-1] > np.pi:
            tmp_dir[j] -= 2*np.pi
    tmp_dir = np.degrees(tmp_dir)
    df["walking_direction"] = tmp_dir
    
    tmp_dir = df["walking_direction_robot"].values
    for j in range(1, len(tmp_dir)):
        if tmp_dir[j-1] > 0 and tmp_dir[j] < 0 and tmp_dir[j-1] - tmp_dir[j] > np.pi:
            tmp_dir[j] += 2*np.pi
        elif tmp_dir[j-1] < 0 and tmp_dir[j] > 0 and tmp_dir[j] - tmp_dir[j-1] > np.pi:
            tmp_dir[j] -= 2*np.pi
    tmp_dir = np.degrees(tmp_dir)
    df["walking_direction_robot"] = tmp_dir
    
    return df

def get_clean_walking_direction_np(df, walking_direction_col):
    X = np.array([[df["seconds"].values[i], df[walking_direction_col].values[i]] for i in range(len(df[walking_direction_col]))])
    cleaned_walking_dir = [x for x in df[walking_direction_col] if not np.isnan(x)]
    # turn NaNs into interpolation
    if np.isnan(X[0][0]):
        X[0][0] = 0
    if np.isnan(X[-1][0]):
        X[-1][0] = 0
    for i in range(1, len(X)):
        if np.isnan(X[i][0]):
            X[i][0] = (X[i-1][0] + X[i+1][0])/2
    X[0][1] = X[1][1]
    i = 1 
    while np.isnan(X[i][1]):
        X[i][1] = cleaned_walking_dir[0]
        i+=1
    i = 1
    while np.isnan(X[-i][1]):
        X[-i][1] = cleaned_walking_dir[-1]
    for i in range(1, len(X)):
        if np.isnan(X[i][1]):
            X[i][1] = (X[i-1][1] + X[i+1][1])/2
    return X

def encode_walking_direction(X, direction_in_cluster_thres=1, cluster_leak_thres=1, starting_leak_allowance=-5):
    walking_direction_np = X[:,1]
    avg_cluster_direction = walking_direction_np[0]
    curr_cluster = 0
    curr_cluster_count = 1
    curr_num_leaked = starting_leak_allowance
    clustering = [0]
    cluster_avgs = []
    leaked_directions_buffer = []
    for i in range(1, len(walking_direction_np)):
        if abs(walking_direction_np[i] - avg_cluster_direction) < direction_in_cluster_thres:
            avg_cluster_direction = (curr_cluster_count*avg_cluster_direction + walking_direction_np[i])/(curr_cluster_count+1)
            curr_cluster_count += 1
            curr_num_leaked = curr_num_leaked-1 if curr_num_leaked > 0 else curr_num_leaked
            if len(leaked_directions_buffer) > 0:
                clustering.extend([curr_cluster]*len(leaked_directions_buffer))
            leaked_directions_buffer = []
            clustering.append(curr_cluster)
        elif curr_num_leaked > cluster_leak_thres:
            cluster_avgs.append(avg_cluster_direction)
            leaked_directions_buffer.append(walking_direction_np[i])
            avg_cluster_direction = sum(leaked_directions_buffer)/len(leaked_directions_buffer)
            curr_cluster_count = 1
            curr_num_leaked = 0
            curr_cluster += 1
            clustering.extend([curr_cluster]*len(leaked_directions_buffer))
            leaked_directions_buffer = []
        else:
            curr_num_leaked += 1
            leaked_directions_buffer.append(walking_direction_np[i])
    if cluster_avgs[-1] != avg_cluster_direction:
        cluster_avgs.append(avg_cluster_direction)
    if len(leaked_directions_buffer) > 0:
        clustering.extend([curr_cluster]*len(leaked_directions_buffer))
    walking_direction_encoded = [cluster_avgs[clustering[i]] for i in range(len(clustering))]
    return walking_direction_encoded

def compute_turnings_legacy(df, walking_direction_encoded, inter_cluster_is_turn_thres = 50, avg_radius_pre_post_turn_limit=40, turning_window_size=20, start_end_buffer=50, verbose=False):
    
    is_turning = [0]*len(walking_direction_encoded)
    for i in range(start_end_buffer+turning_window_size, (len(walking_direction_encoded)-start_end_buffer-turning_window_size)-1):
        if abs(walking_direction_encoded[i] - walking_direction_encoded[i+1]) > inter_cluster_is_turn_thres:
            is_turning[i-turning_window_size:i+2+turning_window_size] = [1]*(turning_window_size*2+2)
            if verbose:
                print(walking_direction_encoded[i], walking_direction_encoded[i+1], df["seconds"][i], i)
    
    # calculating the avg direction between the computed turns
    curr_avg_window_count = 1
    avgs_between = []
    starting_indices = [0]
    curr_avg = walking_direction_encoded[0]
    last_avg = walking_direction_encoded[0]
    turning = False
    for i in range(1, len(walking_direction_encoded)-1):
        if is_turning[i] == 1 and not turning:
            avgs_between.append(curr_avg%360)
            turning = True
        elif is_turning[i] != 1:
            if turning:
                curr_avg = walking_direction_encoded[i]
                last_avg = walking_direction_encoded[i]
                starting_indices.append(i)
                curr_avg_window_count = 1
                turning = False
            else:
                if curr_avg_window_count < avg_radius_pre_post_turn_limit:
                    curr_avg = (curr_avg*curr_avg_window_count + walking_direction_encoded[i])/(curr_avg_window_count+1)
                    curr_avg_window_count += 1
                    last_avg = walking_direction_encoded[i]
                else:
                    curr_avg = (curr_avg*avg_radius_pre_post_turn_limit - last_avg + walking_direction_encoded[i])/avg_radius_pre_post_turn_limit
    avgs_between.append(curr_avg%360)
    if verbose:
        print(avgs_between)

    for i in range(1, len(avgs_between)):
        if abs(avgs_between[i] - avgs_between[i-1]) < inter_cluster_is_turn_thres:
            if verbose:
                print("*DEL", avgs_between[i-1], avgs_between[i], df["seconds"][starting_indices[i]], starting_indices[i])
            is_turning[starting_indices[i-1]:starting_indices[i]] = [0]*(starting_indices[i]-starting_indices[i-1])
        else:
            if verbose:
                print(avgs_between[i-1], avgs_between[i], df["seconds"][starting_indices[i]], starting_indices[i])
            
    return is_turning

def compute_basic_turnings(df, walking_direction_encoded, inter_cluster_is_turn_thres = 50, turning_window_size=20, start_end_buffer=50, verbose=False):
    is_turning = [0]*len(walking_direction_encoded)
    for i in range(start_end_buffer+turning_window_size, (len(walking_direction_encoded)-start_end_buffer-turning_window_size)-1):
        if abs(walking_direction_encoded[i] - walking_direction_encoded[i+1]) > inter_cluster_is_turn_thres:
            is_turning[i-turning_window_size:i+2+turning_window_size] = [1]*(turning_window_size*2+2)
            if verbose:
                print(walking_direction_encoded[i], walking_direction_encoded[i+1], df["seconds"][i], i)
    return is_turning

def remove_non_decisive_turns(df, walking_direction_encoded, is_turning, avg_radius_pre_post_turn_limit=40, ignore_turn_pre_post_avgs_diff_thres=50, verbose=False):
    # calculating the start and end indices of no turn zones
    starting_indices = []
    ending_indices = []
    if is_turning[0] == 0:
        starting_indices.append(0)
    for i in range(1, len(walking_direction_encoded)-1):
        if is_turning[i] == 0 and is_turning[i+1] == 1:
            ending_indices.append(i)
        elif is_turning[i] == 1 and is_turning[i+1] == 0:
            starting_indices.append(i)
    ending_indices.append(len(walking_direction_encoded)-1)
    assert len(starting_indices) == len(ending_indices)
    num_turns = len(starting_indices) - 1
    
    # calculating the avg direction before and after each turn and removing turns that are not decisive
    for i in range(num_turns):
        before_start_index, before_end_index = starting_indices[i], ending_indices[i]
        after_start_index, after_end_index = starting_indices[i+1], ending_indices[i+1]
        if before_end_index - before_start_index < avg_radius_pre_post_turn_limit:
            before_avg = np.mean(walking_direction_encoded[before_start_index:before_end_index+1])%360
        else:
            before_avg = np.mean(walking_direction_encoded[before_end_index-avg_radius_pre_post_turn_limit:before_end_index+1])%360
        if after_end_index - after_start_index < avg_radius_pre_post_turn_limit:
            after_avg = np.mean(walking_direction_encoded[after_start_index:after_end_index+1])%360
        else:
            after_avg = np.mean(walking_direction_encoded[after_start_index:after_start_index+avg_radius_pre_post_turn_limit+1])%360
        if abs(before_avg - after_avg) < ignore_turn_pre_post_avgs_diff_thres:
            if verbose:
                print("*DEL", before_avg, after_avg, df["seconds"][starting_indices[i]], starting_indices[i])
            is_turning[starting_indices[i]:starting_indices[i+1]] = [0]*(starting_indices[i+1]-starting_indices[i])
        else:
            if verbose:
                print(before_avg, after_avg, df["seconds"][starting_indices[i]], starting_indices[i])
    return is_turning

def get_lmrs(is_turning, verbose=False):
    middles = []
    lefts = []
    rights = []
    for i in range(len(is_turning)):
        if is_turning[i] == 1:
            if is_turning[i-1] == 0 and is_turning[i+1] == 1:
                lefts.append(i)
                if verbose:
                    print("left: ", i)
            elif is_turning[i-1] == 1 and is_turning[i+1] == 0:
                rights.append(i)
                middles.append((i+lefts[-1])//2)
                if verbose:
                    print("middle", (i+lefts[-1])//2)
                    print("right: ", i)
    return lefts, middles, rights

def plot_main_map_with_turns(df, pre, lmrs, lmrs_robot, black_radius=2, black_mod=50, blue_radius=2, start_thres=20, turnings=True, green_start=False, red_end=True, black_marks=True, blue_marks=True, grey_marks=True, verbose=False, savefig=True):
    if DEBUG:
        return
    n = len(df["GPS1 (Long.) [deg]"])
    
    plt.clf()
    plt.plot(df["GPS1 (Long.) [deg]"], df["GPS1 (Lat.) [deg]"], color="coral", label="participant_path", alpha=0.2)
    plt.plot(df["GPS2 (Long.) [deg]"], df["GPS2 (Lat.) [deg]"], color="green", label="robot_path", alpha=0.2)
    
    if turnings:
        plt.plot(df.where(df["is_turning"])["GPS1 (Long.) [deg]"], df.where(df["is_turning"])["GPS1 (Lat.) [deg]"], color="coral", label="participant_turning")
        plt.plot(df.where(df["is_turning_robot"])["GPS2 (Long.) [deg]"], df.where(df["is_turning_robot"])["GPS2 (Lat.) [deg]"], color="green", label="robot_turning")

    if green_start:
        plt.plot(df["GPS1 (Long.) [deg]"][:start_thres], df["GPS1 (Lat.) [deg]"][:start_thres], color="green")
        plt.plot(df["GPS2 (Long.) [deg]"][:start_thres], df["GPS2 (Lat.) [deg]"][:start_thres], color="green")
        
    if red_end:
        plt.plot(df["GPS1 (Long.) [deg]"][n-start_thres:], df["GPS1 (Lat.) [deg]"][n-start_thres:], color="red")
        plt.plot(df["GPS2 (Long.) [deg]"][n-start_thres:], df["GPS2 (Lat.) [deg]"][n-start_thres:], color="red")
        
    if black_marks:
        black_indices = []
        for i in range(start_thres+black_radius+1,n-start_thres-black_radius-1):
            if i%black_mod == 0:
                black_indices += list(range(i-black_radius, i+black_radius+1))
        df["is_black"] = [i in black_indices for i in range(n)]
        plt.plot(df.where(df["is_black"])["GPS1 (Long.) [deg]"], df.where(df["is_black"])["GPS1 (Lat.) [deg]"], color="black")
        plt.plot(df.where(df["is_black"])["GPS2 (Long.) [deg]"], df.where(df["is_black"])["GPS2 (Lat.) [deg]"], color="black")
        
    if blue_marks:
        n = len(df["GPS1 (Long.) [deg]"])
        turn_indices = []
        lefts, middles, rights = lmrs
        for i in range(start_thres+blue_radius+1,n-start_thres-blue_radius-1):
            if i in (middles+lefts+rights):
                turn_indices += list(range(i-blue_radius, i+blue_radius+1))
        df["is_blue"] = [i in turn_indices for i in range(n)]
        turn_indices_robot = []
        lefts_robot, middles_robot, rights_robot = lmrs_robot
        for i in range(start_thres+blue_radius+1,n-start_thres-blue_radius-1):
            if i in (middles_robot+lefts_robot+rights_robot):
                turn_indices_robot += list(range(i-blue_radius, i+blue_radius+1))
        df["is_blue_robot"] = [i in turn_indices_robot for i in range(n)]
        plt.plot(df.where(df["is_blue"])["GPS1 (Long.) [deg]"], df.where(df["is_blue"])["GPS1 (Lat.) [deg]"], color="blue")
        plt.plot(df.where(df["is_blue_robot"])["GPS2 (Long.) [deg]"], df.where(df["is_blue_robot"])["GPS2 (Lat.) [deg]"], color="blue")
        
    plt.xlabel("Long. [deg]", fontsize=8)  # Adjust the font size
    plt.ylabel("Lat. [deg]", fontsize=8)   # Adjust the font size
    plt.title("Turning Map")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
    plt.legend()
    if savefig:
        plt.savefig(f"{pre}turning_paths.png")
    else:
        plt.show()
        plt.clf()

def plot_advanced(df, pre, only_person=False):
    if DEBUG:
        return
    # plotting paths
    plt.clf()
    plt.plot(df["GPS1 (Long.) [deg]"], df["GPS1 (Lat.) [deg]"], color="coral", label="participant_path")
    if not only_person:
        plt.plot(df["GPS2 (Long.) [deg]"], df["GPS2 (Lat.) [deg]"], color="blue", label="robot_path")
    plt.xlabel("Long. [deg]", fontsize=8)  # Adjust the font size
    plt.ylabel("Lat. [deg]", fontsize=8)   # Adjust the font size
    plt.title("Walking Paths")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
    plt.legend()
    plt.savefig(f"{pre}paths.png")
    
    # plot walking directions of participant and robot
    # line_plot(df, "seconds", ["walking_direction", "walking_direction_robot"], "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction.png")
    line_plot(df, "seconds", "walking_direction", "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction_person.png")
    if not only_person:
        line_plot(df, "seconds", "walking_direction", "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction_robot.png")

    # plot encoded walking directions of participant and robot
    line_plot(df, "seconds", ["walking_direction","walking_direction_encoded"], "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction_encoded.png")
    if not only_person:
        line_plot(df, "seconds", ["walking_direction_robot","walking_direction_encoded_robot"], "Walking Directions", "Time (s)", "Walking Direction (deg)", f"{pre}walking_direction_encoded_robot.png")

    # plot turning
    # line_plot(df, "seconds", ["is_turning", "is_turning_robot"], "Turning", "Time (s)", "Turning", f"{pre}is_turning.png")
    line_plot(df, "seconds", "is_turning", "Turning", "Time (s)", "Turning", f"{pre}is_turning_person.png")
    if not only_person:
        line_plot(df, "seconds", "is_turning", "Turning", "Time (s)", "Turning", f"{pre}is_turning_robot.png")
    
def discretize_turnings(df):
    thres = 0
    df["is_turning"] = df["is_turning"].apply(lambda x: abs(x) > thres)
    df["is_turning_robot"] = df["is_turning_robot"].apply(lambda x: abs(x) > thres)
    return df

def plot_single_turn(df, pre, path_num, turn_num, lmrs, person=True, verbose=True, **plot_options):
    n = len(df["GPS1 (Long.) [deg]"])
    
    lefts, _, rights = lmrs
    single_turn_indices = list(range(lefts[turn_num], rights[turn_num]+1))
    df["curr_single_turn"] = [i in single_turn_indices for i in range(n)]
    
    if verbose:
        print(f'{"Person" if person else "Robot"} - Path {path_num+1} Turn {turn_num+1}')
        print("single turn seconds: ", df["seconds"][df["curr_single_turn"]].to_list())
        
    assert df["seconds"][single_turn_indices].to_list() == df["seconds"][df["curr_single_turn"]].to_list()
    
    x, y = df["walking_direction"][single_turn_indices], df["walking_direction_robot"][single_turn_indices]
    walking_direction_base_corr, walking_direction_lag, walking_direction_lagged_corr = get_lag(x.values, y.values, title="Walking Direction", verbose=verbose)
    walking_direction_dtw = get_dtw(x.values, y.values, title="walking_direction", pre=pre)
    x, y = df["GPS1 (2D) [m/s]"][single_turn_indices], df["GPS2 (2D) [m/s]"][single_turn_indices]
    speeds_base_corr, speeds_lag, speeds_lagged_corr = get_lag(x.values, y.values, title="Speed", verbose=verbose)
    speeds_dtw = get_dtw(x.values, y.values, title="speeds", pre=pre)
    
    ### calculate walking direction difference ###
    walking_direction = df["walking_direction"].values
    walking_direction_robot = df["walking_direction_robot"].values
    walking_direction_difference = [abs(walking_direction[i]%360 - walking_direction_robot[i]%360) for i in range(len(df["walking_direction"]))]
    for i in range(1, len(walking_direction_difference)):
        if walking_direction_difference[i] > 180:
            walking_direction_difference[i] = 360 - walking_direction_difference[i]
    df["walking_direction_difference"] = walking_direction_difference
    
    mean_distance = np.mean(df["distance"][single_turn_indices])
    mean_speed_difference = np.mean(df["speed_difference"][single_turn_indices])
    mean_walking_direction_difference = np.mean(df["walking_direction_difference"][single_turn_indices])
    mean_pace_asymmetry = np.mean(df["pace_difference"][single_turn_indices])
    
    if not DEBUG:
        if plot_options["single_turn_paths"]:
            plt.clf()
            plt.plot(df.where(df["curr_single_turn"])["GPS1 (Long.) [deg]"], df.where(df["curr_single_turn"])["GPS1 (Lat.) [deg]"], color="coral", label="participant_turning")
            plt.plot(df.where(df["curr_single_turn"])["GPS2 (Long.) [deg]"], df.where(df["curr_single_turn"])["GPS2 (Lat.) [deg]"], color="green", label="robot_during")
            start_thres = 1
            black_indices = [i for i in single_turn_indices if i > start_thres + single_turn_indices[0] and i < single_turn_indices[-1] and i%10 in [0,1]]
            grey_indices = [i for i in single_turn_indices if i > start_thres + single_turn_indices[0] and i < single_turn_indices[-1] and i%10 not in [0,1] and i%5 in [0,1]]
            df["is_black_curr_single_turn"] = [i in black_indices for i in range(n)]
            df["is_grey_curr_single_turn"] = [i in grey_indices for i in range(n)]
            if verbose:
                print("Black indices seconds: ", df["seconds"][df["is_black_curr_single_turn"]].to_list())
                print("Grey indices seconds: ", df["seconds"][df["is_grey_curr_single_turn"]].to_list())
            plt.plot(df.where(df["is_black_curr_single_turn"])["GPS1 (Long.) [deg]"], df.where(df["is_black_curr_single_turn"])["GPS1 (Lat.) [deg]"], color="black")
            plt.plot(df.where(df["is_grey_curr_single_turn"])["GPS1 (Long.) [deg]"], df.where(df["is_grey_curr_single_turn"])["GPS1 (Lat.) [deg]"], color="brown")
            plt.plot(df.where(df["is_black_curr_single_turn"])["GPS2 (Long.) [deg]"], df.where(df["is_black_curr_single_turn"])["GPS2 (Lat.) [deg]"], color="black")
            plt.plot(df.where(df["is_grey_curr_single_turn"])["GPS2 (Long.) [deg]"], df.where(df["is_grey_curr_single_turn"])["GPS2 (Lat.) [deg]"], color="brown")
            plt.plot(df["GPS1 (Long.) [deg]"][single_turn_indices[-1]-start_thres:single_turn_indices[-1]+1], df["GPS1 (Lat.) [deg]"][single_turn_indices[-1]-start_thres:single_turn_indices[-1]+1], color="red")
            plt.plot(df["GPS2 (Long.) [deg]"][single_turn_indices[-1]-start_thres:single_turn_indices[-1]+1], df["GPS2 (Lat.) [deg]"][single_turn_indices[-1]-start_thres:single_turn_indices[-1]+1], color="red")
            plt.xlabel("Long. [deg]", fontsize=8)  # Adjust the font size
            plt.ylabel("Lat. [deg]", fontsize=8)   # Adjust the font size
            plt.title("Single Turn Map")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}paths.png")

        if plot_options["distance"]:
            plt.clf()
            plt.plot(df["seconds"][df["curr_single_turn"]], df["distance"][single_turn_indices], color="coral", label="distance")
            plt.xlabel("Time (s)", fontsize=8)  # Adjust the font size
            plt.ylabel("Distance (m)", fontsize=8)   # Adjust the font size
            plt.title(f"Run {path_num+1} Turn {turn_num+1} - Distance \nMean: {round(mean_distance,3)}m")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}distance.png")

        if plot_options["speed_difference"]:
            plt.clf()
            plt.plot(df["seconds"][df["curr_single_turn"]], df["speed_difference"][single_turn_indices], color="coral", label="speed_difference")
            plt.xlabel("Time (s)", fontsize=8)  # Adjust the font size
            plt.ylabel("Speed Difference (m/s)", fontsize=8)   # Adjust the font size
            plt.title(f"Run {path_num+1} Turn {turn_num+1} - Speed Difference \nMean: {round(mean_speed_difference,3)} m/s")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}speed_difference.png")
        
        if plot_options["walking_direction_difference"]:
            # plot walking direction difference over the single turn
            plt.clf()
            plt.plot(df["seconds"][df["curr_single_turn"]], df["walking_direction_difference"][single_turn_indices], color="coral", label="walking_direction_difference")
            plt.xlabel("Time (s)", fontsize=8)  # Adjust the font size
            plt.ylabel("Walking Direction Difference (deg)", fontsize=8)   # Adjust the font size
            # degree symbol is \u00b0
            plt.title(f"Run {path_num+1} Turn {turn_num+1} - Walking Direction Difference \nMean: {round(mean_walking_direction_difference,3)}\u00b0")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}walking_direction_difference.png")
        
        if plot_options["speeds"]:
            plt.clf()
            plt.plot(df["seconds"][df["curr_single_turn"]], df["GPS1 (2D) [m/s]"][single_turn_indices], color="coral", label="speed_person")
            plt.plot(df["seconds"][df["curr_single_turn"]], df["GPS2 (2D) [m/s]"][single_turn_indices], color="green", label="speed_robot")
            plt.xlabel("Time (s)", fontsize=8)  # Adjust the font size
            plt.ylabel("Speed (m/s)", fontsize=8)   # Adjust the font size
            plt.title(f'{"Person" if person else "Robot"} - Run {path_num+1} Turn {turn_num+1} - Speeds \nLag: {speeds_lag}({">" if abs(speeds_lag)==speeds_lag else "<"}{round(abs(speeds_lag)*0.2,3)}s), Corr: Base={round(speeds_base_corr,3)}, Lagged={round(speeds_lagged_corr,3)}, dtw={round(speeds_dtw,3)}')
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}speeds.png")
        
        if plot_options["walking_directions"]:
            plt.clf()
            plt.plot(df["seconds"][df["curr_single_turn"]], df["walking_direction"][single_turn_indices], color="coral", label="walking_direction_person")
            plt.plot(df["seconds"][df["curr_single_turn"]], df["walking_direction_robot"][single_turn_indices], color="green", label="walking_direction_robot")
            plt.xlabel("Time (s)", fontsize=8)  # Adjust the font size
            plt.ylabel("Walking Direction (deg)", fontsize=8)   # Adjust the font size
            plt.title(f'{"Person" if person else "Robot"} - Run {path_num+1} Turn {turn_num+1} - Walking Directions \nLag: {walking_direction_lag}({">" if abs(walking_direction_lag)==walking_direction_lag else "<"}{round(abs(walking_direction_lag)*0.2,3)}s), Corr: Base={round(walking_direction_base_corr,3)}, Lagged={round(walking_direction_lagged_corr,3)}, dtw={round(walking_direction_dtw,3)}')
            plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="lightgray")
            plt.legend()
            plt.savefig(f"{pre}walking_directions.png")

    if verbose:
        print(f"Mean distance over single turn: {round(mean_distance,3)}")
        print(f"Mean speed difference over single turn: {round(mean_speed_difference,3)}")
        print(f"Mean walking direction difference over single turn: {round(mean_walking_direction_difference,3)}")
        print()
        
    return {"start_idx":lefts[turn_num], 
            "end_idx":rights[turn_num]+1,
            "walking_direction_lag": walking_direction_lag, 
            "walking_direction_base_corr": walking_direction_base_corr, 
            "walking_direction_lagged_corr": walking_direction_lagged_corr, 
            "walking_direction_dtw": walking_direction_dtw,
            "speeds_lag": speeds_lag, 
            "speeds_base_corr": speeds_base_corr, 
            "speeds_lagged_corr": speeds_lagged_corr,
            "speeds_dtw": speeds_dtw,
            "mean_distance" : mean_distance,
            "mean_speed_difference" : mean_speed_difference,
            "mean_walking_direction_difference" : mean_walking_direction_difference,
            "mean_pace_asymmetry" : mean_pace_asymmetry}
    
def plot_all_turns(df, turns_dir, path_num, lmrs, person=True, save_to_csv=True, verbose=True, 
                   plot_options={"single_turn_paths": True, 
                                 "distance": True, "speed_difference": False, "walking_direction_difference": False, 
                                 "speeds": True, "walking_directions": True}):
    if DEBUG:
        plot_options["single_turn_paths"] = False

    lefts, _, rights = lmrs
    path_results_dict = {}
    run_dir = turns_dir + f'{"person" if person else "robot"}/run_{path_num+1}/'
    create_folder(run_dir, instead=False)
    
    for turn_num in range(min(len(lefts), len(rights))):
        pre = run_dir + f"turn_{turn_num+1}/"
        create_folder(pre, instead=False)
        single_turn_res = plot_single_turn(df, pre, path_num, turn_num, lmrs, person=person, verbose=verbose, **plot_options)
        for k in single_turn_res.keys():
            if not k in path_results_dict:
                path_results_dict[k] = []
            path_results_dict[k].append(single_turn_res[k])
            
    if save_to_csv:
        path_results_df = pd.DataFrame(path_results_dict)
        path_results_df.to_csv(run_dir + "lags_corrs_means.csv", index=True)
    
    return path_results_dict
        
def get_lag(x, y, title="", verbose=True):
    x, y = np.array(x), np.array(y)
    x = (x - np.mean(x))/np.std(x)
    y = (y - np.mean(y))/np.std(y)
    correlation = signal.correlate(x, y, mode="full",)
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    base_idx = [i for i, l in enumerate(lags) if l == 0][0]
    std_base_corr = correlation[base_idx]/(np.linalg.norm(x)*np.linalg.norm(y))
    # std_base_corr = get_cosine_sim(x, y)
    std_lagged_corr = np.max(correlation)/(np.linalg.norm(x)*np.linalg.norm(y))
    # if lag > 0:
    #     start_x, end_x = lag, len(x)
    #     start_y, end_y = 0, len(y)-lag
    # else:
    #     start_x, end_x = 0, len(x)+lag
    #     start_y, end_y = -lag, len(y)
    # std_lagged_corr = get_cosine_sim(x[start_x:end_x], y[start_y:end_y])
    if verbose:
        print(title + f" Base Correlation: {round(std_base_corr,3)}")
        print(title + f" Lag: {lag}")
        print(title + f" Lagged Correlation: {round(std_lagged_corr,3)}")
    return std_base_corr, lag, std_lagged_corr

def get_manual_lag(x, y, title="", verbose=True):
    N = max(len(x), len(y))
    correlation = np.array([get_std_corr(x, y, -k+N-1, (len(x)-1)-k+N-1) for k in range((len(y)+len(x)-2)+1)])
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    assert len(correlation) == len(lags)
    lag = lags[np.argmax(correlation)]
    base_idx = [i for i, l in enumerate(lags) if l == 0][0]
    std_base_corr = correlation[base_idx]
    std_lagged_corr = np.max(correlation)
    if verbose:
        print(title + f" Base Correlation: {round(std_base_corr,3)}")
        print(title + f" Lag: {lag}")
        print(title + f" Lagged Correlation: {round(std_lagged_corr,3)}")
    return std_base_corr, lag, std_lagged_corr

def get_std_corr(x, y, y_start, y_end):
    y_lagged = [y[i] if i < len(y) and i >= 0 else 0 for i in range(y_start, y_end+1)]
    
    return np.dot(x, y_lagged)/(np.linalg.norm(x)*np.linalg.norm(y_lagged))

def get_dtw(x, y, title, pre):
    x, y = np.array(x), np.array(y)
    x = (x - np.mean(x))/np.std(x)
    y = (y - np.mean(y))/np.std(y)
    # l2_norm = lambda x, y: (x-y)**2
    # d = dtw.dtw(x, y, dist=l2_norm)
    d = dtw.dtw(x, y)
    dtw.dtwPlotTwoWay(d, x, y)
    plt.savefig(f"{pre}{title}_dtw.png")
    return d.distance
    
def get_pearson_corr(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    return np.sum((x-x_mean)*(y-y_mean))/(len(x)*x_std*y_std)

def get_cosine_sim(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def normalized_cross_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    x_std = np.std(x)
    y_std = np.std(y)
    def pearson(x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return np.sum((x-x_mean)*(y-y_mean))/(x_std*y_std)

def set_debug(debug):
    global DEBUG
    DEBUG = debug

# DEBUG = True