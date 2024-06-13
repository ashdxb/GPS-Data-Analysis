import pandas as pd
import numpy as np
from geopy.distance import geodesic
import math
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import shutil
from sklearn.preprocessing import KBinsDiscretizer

def line_plot(df, x, y, title, xlabel, ylabel, save_path):
    plt.clf()
    if type(y) != str:
        colors = ['coral', 'blue']
        labels = ['participant', 'robot']
        for i, col in enumerate(y):
            sns.lineplot(x=x, y=col, color=colors[i], data=df, label=labels[i])
        plt.legend()
    else:
        sns.lineplot(x=x, y=y, color='green', data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)

for participant_id in [407,408]+[i for i in range(2101,2110)]+[2111]:
    print(f'participant {participant_id}')
    files = sorted(os.listdir(f'./{participant_id}/'))
    # find indexes where the filename contains the string 'robot'
    robot_indexes = [i for i, s in enumerate(files) if 'robot' in s]
    robot_files = sorted([files[i] for i in robot_indexes])
    participant_indexes = [i for i, s in enumerate(files) if 'robot' not in s]
    participant_files = sorted([files[i] for i in participant_indexes])
    for i in range(0, 3):
        # person_csv_file is the i'th file in the participant's folder
        person_csv_file = f'./{participant_id}/{participant_files[i]}'
        # robot_csv_file is the j'th file in the participant's folder
        robot_csv_file = f'./{participant_id}/{robot_files[i]}'
        print(person_csv_file, robot_csv_file)

    # read arg[0] as person csv file
        # read arg[1] as robot csv file
        # person_csv_file = sys.argv[1]
        # robot_csv_file = sys.argv[2]

        # get parent directory from arg[2]
        parent_dir = f'./{participant_id}'

        pre = str(participant_id)
        pre = parent_dir + '/results/' + pre + '_' + str(i+1) + '_'

        # create images folder if it doesn't exist
        import os
        if not os.path.exists(parent_dir + '/results/'):
            os.makedirs(parent_dir + '/results/')
        # elif i == 0:
        #     shutil.rmtree(parent_dir + '/results/', ignore_errors=True)
        #     os.makedirs(parent_dir + '/results/')

        rmse_results = []

        csv1 = pd.read_csv(person_csv_file) #person GPS
        csv2 = pd.read_csv(robot_csv_file) # robot GPS

        csv1['date'] = pd.to_datetime(csv1['date']).dt.tz_localize(None)
        csv2['date'] = pd.to_datetime(csv2['date']).dt.tz_localize(None)

        # Find the minimum date common to both CSV files
        min_date = max(csv1['date'].min(), csv2['date'].min())

        csv1 = csv1[csv1['date'] >= min_date]
        csv2 = csv2[csv2['date'] >= min_date]

        # Extract the required columns
        csv1_data = csv1[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]
        csv2_data = csv2[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]

        csv1_data.columns = ['date', 'GPS1 (Lat.) [deg]', 'GPS1 (Long.) [deg]', 'GPS1 (2D) [m/s]']
        csv2_data.columns = ['date', 'GPS2 (Lat.) [deg]', 'GPS2 (Long.) [deg]', 'GPS2 (2D) [m/s]']

        # Merge the dataframes based on the 'date' column
        merged_data = pd.merge(csv1_data, csv2_data, on='date')
        df = merged_data

        # find first index where robot moves
        threshold = 0.1
        first_move_index = 0
        for j in range(len(df['GPS2 (2D) [m/s]'])):
            if df.loc[j,'GPS2 (2D) [m/s]'] > threshold:
                first_move_index = j
                break

        # remove first_move_index rows from df
        df = df.iloc[first_move_index:]

        # find the minimum length of the two dataframes
        min_length = min(len(df[col]) for col in df.columns)
        df = df.iloc[:min_length]

        # Calculate the difference in milliseconds between each row and the first row
        df['date'] = pd.to_datetime(df['date'])
        first_date = df.iloc[0]['date']
        df['milliseconds'] = (df['date'] - first_date).dt.total_seconds() * 1000
        df['seconds'] = (df['date'] - first_date).dt.total_seconds()

        # calculate difference in milliseconds between each row and the previous row
        df['diff_milliseconds'] = df['milliseconds'].diff()
        df['diff_seconds'] = df['seconds'].diff()

        # # distance from robot to participant and target
        # df['distance'] = df.apply(lambda row: geodesic((row['GPS1 (Lat.) [deg]'], row['GPS1 (Long.) [deg]']), (row['GPS2 (Lat.) [deg]'], row['GPS2 (Long.) [deg]'])).meters, axis=1)
        # df['distance_target'] = df['distance'].apply(lambda x: x if x > 1 else 0)

        # # calculate total distance RMSE
        # distance_rmse = math.sqrt(np.sum((df['distance_target']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
        # print(f"distance target RMSE is {distance_rmse}")
        # rmse_results.append(distance_rmse)

        # Plotting the graph with two lines representing participant's path and robot's path
        # plt.clf()
        # plt.plot(df['GPS1 (Long.) [deg]'], df['GPS1 (Lat.) [deg]'], color='coral', label='participant_path')
        # plt.plot(df['GPS2 (Long.) [deg]'], df['GPS2 (Lat.) [deg]'], color='blue', label='robot_path')
        # plt.xlabel('Long. [deg]', fontsize=8)  # Adjust the font size
        # plt.ylabel('Lat. [deg]', fontsize=8)   # Adjust the font size
        # plt.title('Walking Paths')
        # plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
        # plt.legend()
        # plt.savefig(f'{pre}paths.png')

        # # plotting distance over time
        # line_plot(df, 'seconds', 'distance', 'Distance Between Robot and Participant', 'Time (s)', 'Distance (m)', f'{pre}distance.png')

        # # plotting speeds over time
        # line_plot(df, 'seconds', ['GPS1 (2D) [m/s]', 'GPS2 (2D) [m/s]'], 'Speed Plot', 'Time (s)', 'Speed (m/s)', f'{pre}speeds.png')

        # df['speed_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]'])

        # # plotting speed difference over time
        # line_plot(df, 'seconds', 'speed_difference', 'Speed Difference', 'Time (s)', 'Speed Difference (m/s)', f'{pre}speed_difference.png')

        # df['pace_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]']) / (df['GPS1 (2D) [m/s]'] + df['GPS2 (2D) [m/s]'])

        # # printing speed difference RMSE
        # speed_difference_rmse = math.sqrt(np.sum((df['speed_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
        # print(f"speed difference RMSE is {speed_difference_rmse}")
        # rmse_results.append(speed_difference_rmse)

        # # plotting pace difference over time
        # line_plot(df, 'seconds', 'pace_difference', 'Pace Asymmetry', 'Time (s)', 'Pace Asymmetry', f'{pre}pace_asymmetry.png')

        # calculate walking direction of participant and robot
        df['walking_direction'] = np.arctan2(df['GPS1 (Lat.) [deg]'].diff(), df['GPS1 (Long.) [deg]'].diff())
        df['walking_direction_robot'] = np.arctan2(df['GPS2 (Lat.) [deg]'].diff(), df['GPS2 (Long.) [deg]'].diff())
        
        # print(df['walking_direction_robot'].values)
        # line_plot(df, 'seconds', ['walking_direction', 'walking_direction_robot'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction_original.png')
        # line_plot(df, 'seconds', 'walking_direction', 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction_original.png')

        
        tmp_dir = df['walking_direction'].values
        for j in range(1, len(tmp_dir)):
            if tmp_dir[j-1] > 0 and tmp_dir[j] < 0 and tmp_dir[j-1] - tmp_dir[j] > np.pi:
                tmp_dir[j] += 2*np.pi
            elif tmp_dir[j-1] < 0 and tmp_dir[j] > 0 and tmp_dir[j] - tmp_dir[j-1] > np.pi:
                tmp_dir[j] -= 2*np.pi
        tmp_dir = np.degrees(tmp_dir)
        df['walking_direction'] = tmp_dir
        
        tmp_dir = df['walking_direction_robot'].values
        for j in range(1, len(tmp_dir)):
            if tmp_dir[j-1] > 0 and tmp_dir[j] < 0 and tmp_dir[j-1] - tmp_dir[j] > np.pi:
                tmp_dir[j] += 2*np.pi
            elif tmp_dir[j-1] < 0 and tmp_dir[j] > 0 and tmp_dir[j] - tmp_dir[j-1] > np.pi:
                tmp_dir[j] -= 2*np.pi
        tmp_dir = np.degrees(tmp_dir)
        df['walking_direction_robot'] = tmp_dir
        # tmp_dir = df['walking_direction_robot'].values
        # for j in range(1, len(tmp_dir)):
        #     if tmp_dir[j-1] > 0 and tmp_dir[j] < 0 and tmp_dir[j-1] - tmp_dir[j] > np.pi:
        #         tmp_dir[j] += 2*np.pi
        #     elif tmp_dir[j-1] < 0 and tmp_dir[j] > 0 and tmp_dir[j] - tmp_dir[j-1] > np.pi:
        #         tmp_dir[j] -= 2*np.pi
                
        # smoothing walking direction of participant and robot
        # df['walking_direction'] = df['walking_direction'].rolling(70).mean()
        # df['walking_direction_robot'] = df['walking_direction_robot'].rolling(70).mean()
        
        # plot walking directions of participant and robot
        # print(df['walking_direction_robot'].values)
        # line_plot(df, 'seconds', ['walking_direction', 'walking_direction_robot'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction.png')
        # line_plot(df, 'seconds', 'walking_direction', 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction.png')

        # X is a np array of entries (seconds[i], walking_direction[i])
        X = np.array([[df['seconds'].values[i], df['walking_direction'].values[i]] for i in range(len(df['walking_direction']))])
        cleaned_walking_dir = [x for x in df['walking_direction'] if not np.isnan(x)]
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
                
                
        X_robot = np.array([[df['seconds'].values[i], df['walking_direction_robot'].values[i]] for i in range(len(df['walking_direction_robot']))])
        cleaned_walking_dir_robot = [x for x in df['walking_direction_robot'] if not np.isnan(x)]
        # turn NaNs into interpolation
        if np.isnan(X_robot[0][0]):
            X_robot[0][0] = 0
        if np.isnan(X_robot[-1][0]):
            X_robot[-1][0] = 0
        for i in range(1, len(X_robot)):
            if np.isnan(X_robot[i][0]):
                X_robot[i][0] = (X_robot[i-1][0] + X_robot[i+1][0])/2
        X_robot[0][1] = X_robot[1][1]
        i = 1
        while np.isnan(X_robot[i][1]):
            X_robot[i][1] = cleaned_walking_dir_robot[0]
            i+=1
        i = 1
        while np.isnan(X_robot[-i][1]):
            X_robot[-i][1] = cleaned_walking_dir_robot[-1]
        for i in range(1, len(X_robot)):
            if np.isnan(X_robot[i][1]):
                X_robot[i][1] = (X_robot[i-1][1] + X_robot[i+1][1])/2
                
        # print(X)
        # grid = X
        # strategy = "kmeans"
        # enc = KBinsDiscretizer(
        #     n_bins=8, encode="ordinal", strategy=strategy, subsample=None
        # )
        # enc.fit(X)
        # transformed = enc.transform(X)
        # # print("The transformed array is"+str(transformed))
        # bin_edges = enc.bin_edges_
        # centroids = [np.mean([x for j, x in zip(range(len(df['walking_direction'])-1), df['walking_direction']) if transformed[j][1] == i]) for i in range(len(bin_edges[0])-1)]
        # # print("The Centroids are"+str(centroids))
        # print("The bin edges are"+str(bin_edges[0]))
        # y = [centroids[int(transformed[i][1])] for i in range(len(transformed))]
        # # print(y)
        # # ys = [centroids[int(transformed[i][0])] for i in range(len(transformed))]
        # # print(df['walking_direction'].values.shape)
        # # print(df['seconds'].values.shape)
        # # grid_encoded = enc.transform(grid).reshape(*df['walking_direction'].values.shape)
        # df['walking_direction_encoded'] = y
        direction_cleaned = X[:,1]
        thres = 1
        avg_dir = direction_cleaned[0]
        lk_thres = 1
        lk = -5
        k = 0
        nk = 1
        clustering = [0]
        avgs = []
        buffer = []
        for i in range(1, len(direction_cleaned)):
            if abs(direction_cleaned[i] - avg_dir) < thres:
                avg_dir = (nk*avg_dir + direction_cleaned[i])/(nk+1)
                nk += 1
                lk = lk-1 if lk > 0 else lk
                if len(buffer) > 0:
                    clustering.extend([k]*len(buffer))
                buffer = []
                clustering.append(k)
            elif lk > lk_thres:
                avgs.append(avg_dir)
                buffer.append(direction_cleaned[i])
                avg_dir = sum(buffer)/len(buffer)
                nk = 1
                lk = 0
                k += 1
                clustering.extend([k]*len(buffer))
                buffer = []
            else:
                lk += 1
                buffer.append(direction_cleaned[i])
        if avgs[-1] != avg_dir:
            avgs.append(avg_dir)
        if len(buffer) > 0:
            clustering.extend([k]*len(buffer))
        walking_direction_encoded = [avgs[clustering[i]] for i in range(len(clustering))]
        # print(walking_direction_encoded)
        df['walking_direction_encoded'] = walking_direction_encoded
        line_plot(df, 'seconds', ['walking_direction','walking_direction_encoded'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction_encoded.png')
        
        thres = 50
        turning_window = 12
        start_end_buffer = 50
        is_turning = [0]*len(walking_direction_encoded)
        for i in range(start_end_buffer+turning_window, (len(walking_direction_encoded)-start_end_buffer-turning_window)-1):
            if abs(walking_direction_encoded[i] - walking_direction_encoded[i+1]) > thres:
                is_turning[i-turning_window:i+2+turning_window] = [1]*(turning_window*2+2)
                # print(walking_direction_encoded[i], walking_direction_encoded[i+1], df['seconds'][i], i)
        df['is_turning'] = is_turning

        middles = []
        lefts = []
        rights = []
        for i in range(len(is_turning)):
            if is_turning[i] == 1:
                if is_turning[i-1] == 0 and is_turning[i+1] == 1:
                    lefts.append(i)
                    print("left: ", i)
                elif is_turning[i-1] == 1 and is_turning[i+1] == 0:
                    rights.append(i)
                    middles.append((i+lefts[-1])//2)
                    print("middle", (i+lefts[-1])//2)
                    print("right: ", i)
        
        direction_cleaned_robot = X_robot[:,1]
        # flip order of entries in direction_cleaned_robot
        # direction_cleaned_robot = direction_cleaned_robot[::-1]
        # thres = 50
        avg_dir = direction_cleaned_robot[0]
        # lk_thres = 6
        lk = -10
        k = 0
        nk = 1
        clustering = [0]
        avgs = []
        buffer = []
        for i in range(1, len(direction_cleaned_robot)):
            if abs(direction_cleaned_robot[i] - avg_dir) < thres:
                avg_dir = (nk*avg_dir + direction_cleaned_robot[i])/(nk+1)
                nk += 1
                lk = lk-1 if lk > 0 else lk
                if len(buffer) > 0:
                    clustering.extend([k]*len(buffer))
                buffer = []
                clustering.append(k)
            elif lk > lk_thres:
                avgs.append(avg_dir)
                buffer.append(direction_cleaned_robot[i])
                avg_dir = sum(buffer)/len(buffer)
                nk = 1
                lk = 0
                k += 1
                clustering.extend([k]*len(buffer))
                buffer = []
            else:
                lk += 1
                buffer.append(direction_cleaned_robot[i])
        if avgs[-1] != avg_dir:
            avgs.append(avg_dir)
        if len(buffer) > 0:
            clustering.extend([k]*len(buffer))
        # flip order of entries in clustering and avgs
        # clustering = clustering[::-1]
        # avgs = avgs[::-1]
        walking_direction_encoded_robot = [avgs[clustering[i]] for i in range(len(clustering))]
        df['walking_direction_encoded_robot'] = [avgs[clustering[i]] for i in range(len(clustering))]
        line_plot(df, 'seconds', ['walking_direction_robot','walking_direction_encoded_robot'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction_encoded_robot.png')
        
        is_turning_robot = [0]*len(walking_direction_encoded_robot)
        for i in range(start_end_buffer+turning_window, (len(walking_direction_encoded_robot)-start_end_buffer-turning_window)-1):
            if abs(walking_direction_encoded_robot[i] - walking_direction_encoded_robot[i+1]) > thres:
                is_turning_robot[i-turning_window:i+2+turning_window] = [1]*(turning_window*2+2)
                # print(walking_direction_encoded_robot[i], walking_direction_encoded_robot[i+1], df['seconds'][i], i)
        df['is_turning_robot'] = is_turning_robot

        middles_robot = []
        lefts_robot = []
        rights_robot = []
        for i in range(len(is_turning_robot)):
            if is_turning_robot[i] == 1:
                if is_turning_robot[i-1] == 0 and is_turning_robot[i+1] == 1:
                    lefts_robot.append(i)
                    print("left: ", i)
                elif is_turning_robot[i-1] == 1 and is_turning_robot[i+1] == 0:
                    rights_robot.append(i)
                    middles_robot.append((i+lefts_robot[-1])//2)
                    print("middle", (i+lefts_robot[-1])//2)
                    print("right: ", i)
                    
        # print([float(x) for x in X[:,0]])
        # print(bin_edges[0][1:-1])
        # size = 2
        # df['is_turning'] = [any([float(x)-size/2<=float(t)<=float(x)+size/2 for x in bin_edges[0][1:-1]]) for t in X[:,0]]
        # line_plot(df, 'seconds', 'is_turning', 'Turning', 'Time (s)', 'Turning', f'{pre}is_turning.png')
        
        # df['walking_direction_autocorr'] = df['walking_direction'].rolling(10).apply(lambda x: x.autocorr(lag=1))
        # df['walking_direction_autocorr_robot'] = df['walking_direction_robot'].rolling(10).apply(lambda x: x.autocorr(lag=1))
        
        # calculate walking direction derivative
        # df['walking_direction_derivative'] = df['walking_direction_encoded'].diff()
        # df['walking_direction_derivative_robot'] = df['walking_direction_encoded_robot'].diff()
        
        
        

        # smoothing walking direction derivatives of participant and robot
        # df['walking_direction_derivative'] = df['walking_direction_derivative'].rolling(10, center=True).mean()
        # df['walking_direction_derivative_robot'] = df['walking_direction_derivative_robot'].rolling(10, center=True).mean()
        
        # plot walking direction derivatives of participant and robot
        # line_plot(df, 'seconds', ['walking_direction_derivative', 'walking_direction_derivative_robot'], 'Walking Direction Derivatives', 'Time (s)', 'Walking Direction Derivative (deg/s)', f'{pre}walking_direction_derivative.png')
        # line_plot(df, 'seconds', 'walking_direction_derivative', 'Walking Direction Derivatives', 'Time (s)', 'Walking Direction Derivative (deg/s)', f'{pre}walking_direction_derivative.png')

        # thres = 0
        # calculate turning
        # df['is_turning'] = df['walking_direction_derivative'].apply(lambda x: 1 if abs(x) > thres else 0)
        # df['is_turning_robot'] = df['walking_direction_derivative_robot'].apply(lambda x: 1 if abs(x) > thres else 0)
        
        # smoothing turning
        # rol = 1
        # df['is_turning'] = df['is_turning'].rolling(rol, center=True).mean()
        # df['is_turning_robot'] = df['is_turning_robot'].rolling(rol, center=True).mean()
        
        # plot turning
        # line_plot(df, 'seconds', ['is_turning', 'is_turning_robot'], 'Turning', 'Time (s)', 'Turning', f'{pre}is_turning.png')
        # line_plot(df, 'seconds', 'is_turning', 'Turning', 'Time (s)', 'Turning', f'{pre}is_turning.png')
        
        thres = 0
        df['is_turning'] = df['is_turning'].apply(lambda x: abs(x) > thres)
        df['is_turning_robot'] = df['is_turning_robot'].apply(lambda x: abs(x) > thres)
        
        
        # Plotting the graph with two lines representing participant's path and robot's path only when turning is true
        plt.clf()
        plt.plot(df['GPS1 (Long.) [deg]'], df['GPS1 (Lat.) [deg]'], color='coral', label='participant_path', alpha=0.2)
        plt.plot(df['GPS2 (Long.) [deg]'], df['GPS2 (Lat.) [deg]'], color='green', label='robot_path', alpha=0.2)
        plt.plot(df.where(df['is_turning'])['GPS1 (Long.) [deg]'], df.where(df['is_turning'])['GPS1 (Lat.) [deg]'], color='coral', label='participant_turning')
        plt.plot(df.where(df['is_turning_robot'])['GPS2 (Long.) [deg]'], df.where(df['is_turning_robot'])['GPS2 (Lat.) [deg]'], color='green', label='robot_turning')
        start_thres = 20
        n = len(df['GPS1 (Long.) [deg]'])
        black_indices = []
        black_radius = 2
        for i in range(start_thres+black_radius+1,n-start_thres-black_radius-1):
            if i%100 == 0:
               black_indices += list(range(i-black_radius, i+black_radius+1))
        grey_indices = []
        for i in range(start_thres+black_radius+1,n-start_thres-black_radius-1):
            if i%50 == 0 and i%100 != 0:
               grey_indices += list(range(i-black_radius, i+black_radius+1))
            
        df['is_black'] = [i in black_indices for i in range(n)]
        df['is_grey'] = [i in grey_indices for i in range(n)]

        turn_indices = []
        blue_radius = 2
        for i in range(start_thres+blue_radius+1,n-start_thres-blue_radius-1):
            if i in (middles+lefts+rights):
               turn_indices += list(range(i-blue_radius, i+blue_radius+1))
        df['is_blue'] = [i in turn_indices for i in range(n)]

        turn_indices_robot = []
        blue_radius = 2
        for i in range(start_thres+blue_radius+1,n-start_thres-blue_radius-1):
            if i in (middles_robot+lefts_robot+rights_robot):
               turn_indices_robot += list(range(i-blue_radius, i+blue_radius+1))
        df['is_blue_robot'] = [i in turn_indices_robot for i in range(n)]

        # plt.plot(df['GPS1 (Long.) [deg]'][:start_thres], df['GPS1 (Lat.) [deg]'][:start_thres], color='green')
        # plt.plot(df['GPS2 (Long.) [deg]'][:start_thres], df['GPS2 (Lat.) [deg]'][:start_thres], color='green')
        plt.plot(df['GPS1 (Long.) [deg]'][n-start_thres:], df['GPS1 (Lat.) [deg]'][n-start_thres:], color='red')
        plt.plot(df['GPS2 (Long.) [deg]'][n-start_thres:], df['GPS2 (Lat.) [deg]'][n-start_thres:], color='red')
        plt.plot(df.where(df['is_black'])['GPS1 (Long.) [deg]'], df.where(df['is_black'])['GPS1 (Lat.) [deg]'], color='black')
        plt.plot(df.where(df['is_black'])['GPS2 (Long.) [deg]'], df.where(df['is_black'])['GPS2 (Lat.) [deg]'], color='black')
        plt.plot(df.where(df['is_grey'])['GPS1 (Long.) [deg]'], df.where(df['is_grey'])['GPS1 (Lat.) [deg]'], color='brown')
        plt.plot(df.where(df['is_grey'])['GPS2 (Long.) [deg]'], df.where(df['is_grey'])['GPS2 (Lat.) [deg]'], color='brown')
        plt.plot(df.where(df['is_blue'])['GPS1 (Long.) [deg]'], df.where(df['is_blue'])['GPS1 (Lat.) [deg]'], color='blue')
        plt.plot(df.where(df['is_blue_robot'])['GPS2 (Long.) [deg]'], df.where(df['is_blue_robot'])['GPS2 (Lat.) [deg]'], color='blue')
        # for black_group in black_indices:
        #     plt.plot(df['GPS1 (Long.) [deg]'][black_group], df['GPS1 (Lat.) [deg]'][black_group], color='black')
        #     # plt.plot(df['GPS2 (Long.) [deg]'][black_group], df['GPS2 (Lat.) [deg]'][black_group], color='black')
        # for blue_index in turn_indices:
        #     plt.plot(df['GPS1 (Long.) [deg]'][blue_index], df['GPS1 (Lat.) [deg]'][blue_index], color='blue')
        #     # plt.plot(df['GPS2 (Long.) [deg]'][grey_index], df['GPS2 (Lat.) [deg]'][grey_index], color='grey')
        plt.xlabel('Long. [deg]', fontsize=8)  # Adjust the font size
        plt.ylabel('Lat. [deg]', fontsize=8)   # Adjust the font size
        plt.title('Turning Map')
        plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
        plt.legend()
        plt.savefig(f'{pre}turning_paths.png')
        print()

    # if participant_id == 408:
    #     break

        # # calculate direction difference (angle between participant and robot)
        # df['direction_difference'] = abs(df['walking_direction'] - df['walking_direction_robot'])

        # # if direction_difference > 180, then subtract 180
        # df['direction_difference'] = np.where(df['direction_difference'] > 180, df['direction_difference'] - 180, df['direction_difference'])

        # # printing direction difference RMSE
        # direction_difference_rmse = math.sqrt(np.sum((df['direction_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
        # print(f"direction difference RMSE is {direction_difference_rmse}")
        # rmse_results.append(direction_difference_rmse)

        # # plot direction difference over time
        # line_plot(df, 'seconds', 'direction_difference', 'Angle of Difference in Walking Direction', 'Time (s)', 'Direction Difference (deg)', f'{pre}direction_difference.png')

        # # save RMSE results to csv
        # cols = ['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse']
        # rmse_results_df = pd.DataFrame([rmse_results], columns=cols)
        # rmse_results_df.to_csv(f'{pre}rmse_results.csv', index=False)


# #########################################################################
# for participant_id in [2110]:
#     print(f'participant {participant_id}')
#     files = sorted(os.listdir(f'./{participant_id}/'))
#     # find indexes where the filename contains the string 'robot'
#     robot_indexes = [i for i, s in enumerate(files) if 'robot' in s]
#     robot_files = sorted([files[i] for i in robot_indexes])
#     participant_indexes = [i for i, s in enumerate(files) if 'robot' not in s]
#     participant_files = sorted([files[i] for i in participant_indexes])
#     for i in [0, 1]:
#         # person_csv_file is the i'th file in the participant's folder
#         person_csv_file = f'./{participant_id}/{participant_files[i]}'
#         # robot_csv_file is the j'th file in the participant's folder
#         robot_csv_file = f'./{participant_id}/{robot_files[i]}'
#         print(person_csv_file, robot_csv_file)

#     # read arg[0] as person csv file
#         # read arg[1] as robot csv file
#         # person_csv_file = sys.argv[1]
#         # robot_csv_file = sys.argv[2]

#         # get parent directory from arg[2]
#         parent_dir = f'./{participant_id}'

#         pre = str(participant_id)
#         k = i+1 if i != 4 else i
#         pre = parent_dir + '/results/' + pre + '_'  + str(k) + '_'

#         # create images folder if it doesn't exist
#         import os
#         if not os.path.exists(parent_dir + '/results/'):
#             os.makedirs(parent_dir + '/results/')
#         elif i == 0:
#             shutil.rmtree(parent_dir + '/results/', ignore_errors=True)
#             os.makedirs(parent_dir + '/results/')

#         rmse_results = []

#         csv1 = pd.read_csv(person_csv_file) #person GPS
#         csv2 = pd.read_csv(robot_csv_file) # robot GPS

#         csv1['date'] = pd.to_datetime(csv1['date']).dt.tz_localize(None)
#         csv2['date'] = pd.to_datetime(csv2['date']).dt.tz_localize(None)

#         # Find the minimum date common to both CSV files
#         min_date = max(csv1['date'].min(), csv2['date'].min())

#         csv1 = csv1[csv1['date'] >= min_date]
#         csv2 = csv2[csv2['date'] >= min_date]

#         # Extract the required columns
#         csv1_data = csv1[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]
#         csv2_data = csv2[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]

#         csv1_data.columns = ['date', 'GPS1 (Lat.) [deg]', 'GPS1 (Long.) [deg]', 'GPS1 (2D) [m/s]']
#         csv2_data.columns = ['date', 'GPS2 (Lat.) [deg]', 'GPS2 (Long.) [deg]', 'GPS2 (2D) [m/s]']

#         # Merge the dataframes based on the 'date' column
#         merged_data = pd.merge(csv1_data, csv2_data, on='date')
#         df = merged_data

#         # find first index where robot moves
#         threshold = 0.1
#         first_move_index = 0
#         for j in range(len(df['GPS2 (2D) [m/s]'])):
#             if df.loc[j,'GPS2 (2D) [m/s]'] > threshold:
#                 first_move_index = j
#                 break

#         # remove first_move_index rows from df
#         df = df.iloc[first_move_index:]

#         # find the minimum length of the two dataframes
#         min_length = min(len(df[col]) for col in df.columns)
#         df = df.iloc[:min_length]

#         # Calculate the difference in milliseconds between each row and the first row
#         df['date'] = pd.to_datetime(df['date'])
#         first_date = df.iloc[0]['date']
#         df['milliseconds'] = (df['date'] - first_date).dt.total_seconds() * 1000
#         df['seconds'] = (df['date'] - first_date).dt.total_seconds()

#         # calculate difference in milliseconds between each row and the previous row
#         df['diff_milliseconds'] = df['milliseconds'].diff()
#         df['diff_seconds'] = df['seconds'].diff()

#         # distance from robot to participant and target
#         df['distance'] = df.apply(lambda row: geodesic((row['GPS1 (Lat.) [deg]'], row['GPS1 (Long.) [deg]']), (row['GPS2 (Lat.) [deg]'], row['GPS2 (Long.) [deg]'])).meters, axis=1)
#         df['distance_target'] = df['distance'].apply(lambda x: x if x > 1 else 0)

#         # calculate total distance RMSE
#         distance_rmse = math.sqrt(np.sum((df['distance_target']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#         print(f"distance target RMSE is {distance_rmse}")
#         rmse_results.append(distance_rmse)

#         # Plotting the graph with two lines representing participant's path and robot's path
#         plt.clf()
#         plt.plot(df['GPS1 (Long.) [deg]'], df['GPS1 (Lat.) [deg]'], color='coral', label='participant_path')
#         plt.plot(df['GPS2 (Long.) [deg]'], df['GPS2 (Lat.) [deg]'], color='blue', label='robot_path')
#         plt.xlabel('Long. [deg]', fontsize=8)  # Adjust the font size
#         plt.ylabel('Lat. [deg]', fontsize=8)   # Adjust the font size
#         plt.title('Walking Paths')
#         plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
#         plt.legend()
#         plt.savefig(f'{pre}paths.png')

#         # plotting distance over time
#         line_plot(df, 'seconds', 'distance', 'Distance Between Robot and Participant', 'Time (s)', 'Distance (m)', f'{pre}distance.png')

#         # plotting speeds over time
#         line_plot(df, 'seconds', ['GPS1 (2D) [m/s]', 'GPS2 (2D) [m/s]'], 'Speed Plot', 'Time (s)', 'Speed (m/s)', f'{pre}speeds.png')

#         df['speed_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]'])

#         # plotting speed difference over time
#         line_plot(df, 'seconds', 'speed_difference', 'Speed Difference', 'Time (s)', 'Speed Difference (m/s)', f'{pre}speed_difference.png')

#         df['pace_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]']) / (df['GPS1 (2D) [m/s]'] + df['GPS2 (2D) [m/s]'])

#         # printing speed difference RMSE
#         speed_difference_rmse = math.sqrt(np.sum((df['speed_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#         print(f"speed difference RMSE is {speed_difference_rmse}")
#         rmse_results.append(speed_difference_rmse)

#         # plotting pace difference over time
#         line_plot(df, 'seconds', 'pace_difference', 'Pace Asymmetry', 'Time (s)', 'Pace Asymmetry', f'{pre}pace_asymmetry.png')

#         # calculate walking direction of participant and robot
#         df['walking_direction'] = np.arctan2(df['GPS1 (Lat.) [deg]'].diff(), df['GPS1 (Long.) [deg]'].diff()) * 180 / np.pi
#         df['walking_direction_robot'] = np.arctan2(df['GPS2 (Lat.) [deg]'].diff(), df['GPS2 (Long.) [deg]'].diff()) * 180 / np.pi

#         # smoothing walking direction of participant and robot
#         df['walking_direction'] = df['walking_direction'].rolling(10).mean()
#         df['walking_direction_robot'] = df['walking_direction_robot'].rolling(10).mean()

#         # plot walking directions of participant and robot
#         line_plot(df, 'seconds', ['walking_direction', 'walking_direction_robot'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction.png')

#         # calculate direction difference (angle between participant and robot)
#         df['direction_difference'] = abs(df['walking_direction'] - df['walking_direction_robot'])

#         # if direction_difference > 180, then subtract 180
#         df['direction_difference'] = np.where(df['direction_difference'] > 180, df['direction_difference'] - 180, df['direction_difference'])

#         # printing direction difference RMSE
#         direction_difference_rmse = math.sqrt(np.sum((df['direction_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#         print(f"direction difference RMSE is {direction_difference_rmse}")
#         rmse_results.append(direction_difference_rmse)

#         # plot direction difference over time
#         line_plot(df, 'seconds', 'direction_difference', 'Angle of Difference in Walking Direction', 'Time (s)', 'Direction Difference (deg)', f'{pre}direction_difference.png')

#         # save RMSE results to csv
#         cols = ['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse']
#         rmse_results_df = pd.DataFrame([rmse_results], columns=cols)
#         rmse_results_df.to_csv(f'{pre}rmse_results.csv', index=False)

#         ##################################################

#         # person_csv_file is the i'th file in the participant's folder
#         person_csv_file_3a = f'./{participant_id}/{participant_files[2]}'
#         # robot_csv_file is the j'th file in the participant's folder
#         robot_csv_file_3a = f'./{participant_id}/{robot_files[2]}'
#         print(person_csv_file_3a, robot_csv_file_3a)

#         # person_csv_file is the i'th file in the participant's folder
#         person_csv_file_3b = f'./{participant_id}/{participant_files[3]}'
#         # robot_csv_file is the j'th file in the participant's folder
#         robot_csv_file_3b = f'./{participant_id}/{robot_files[3]}'
#         print(person_csv_file_3b, robot_csv_file_3b)


#     # read arg[0] as person csv file
#         # read arg[1] as robot csv file
#         # person_csv_file = sys.argv[1]
#         # robot_csv_file = sys.argv[2]

#     # get parent directory from arg[2]
#     parent_dir = f'./{participant_id}'

#     pre = str(participant_id)
#     pre = parent_dir + '/results/' + pre + '_'  + str(3) + '_'

#     # create images folder if it doesn't exist
#     import os
#     if not os.path.exists(parent_dir + '/results/'):
#         os.makedirs(parent_dir + '/results/')
#     elif i == 0:
#         # remove foSlder and its contents
#         shutil.rmtree(parent_dir + '/results/', ignore_errors=True)
#         os.makedirs(parent_dir + '/results/')

#     rmse_results = []

#     csv1a = pd.read_csv(person_csv_file_3a) #person GPS
#     csv2a = pd.read_csv(robot_csv_file_3a) # robot GPS

#     csv1a['date'] = pd.to_datetime(csv1a['date']).dt.tz_localize(None)
#     csv2a['date'] = pd.to_datetime(csv2a['date']).dt.tz_localize(None)

#     # Find the minimum date common to both CSV files
#     min_date_3a = max(csv1a['date'].min(), csv2a['date'].min())

#     csv1a = csv1a[csv1a['date'] >= min_date_3a]
#     csv2a = csv2a[csv2a['date'] >= min_date_3a]

#     # Extract the required columns
#     csv1a_data = csv1a[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]
#     csv2a_data = csv2a[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]

#     csv1a_data.columns = ['date', 'GPS1 (Lat.) [deg]', 'GPS1 (Long.) [deg]', 'GPS1 (2D) [m/s]']
#     csv2a_data.columns = ['date', 'GPS2 (Lat.) [deg]', 'GPS2 (Long.) [deg]', 'GPS2 (2D) [m/s]']

#     # Merge the dataframes based on the 'date' column
#     merged_data_3a = pd.merge(csv1a_data, csv2a_data, on='date')
#     df_3a = merged_data_3a

#     csv1b = pd.read_csv(person_csv_file_3b) #person GPS
#     csv2b = pd.read_csv(robot_csv_file_3b) # robot GPS

#     csv1b['date'] = pd.to_datetime(csv1b['date']).dt.tz_localize(None)
#     csv2b['date'] = pd.to_datetime(csv2b['date']).dt.tz_localize(None)

#     # Find the minimum date common to both CSV files
#     min_date_3b = max(csv1b['date'].min(), csv2b['date'].min())

#     csv1b = csv1b[csv1b['date'] >= min_date_3b]
#     csv2b = csv2b[csv2b['date'] >= min_date_3b]

#     # Extract the required columns
#     csv1b_data = csv1b[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]
#     csv2b_data = csv2b[['date', 'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (2D) [m/s]']]

#     csv1b_data.columns = ['date', 'GPS1 (Lat.) [deg]', 'GPS1 (Long.) [deg]', 'GPS1 (2D) [m/s]']
#     csv2b_data.columns = ['date', 'GPS2 (Lat.) [deg]', 'GPS2 (Long.) [deg]', 'GPS2 (2D) [m/s]']

#     # Merge the dataframes based on the 'date' column
#     merged_data_3b = pd.merge(csv1b_data, csv2b_data, on='date')
#     df_3b = merged_data_3b

#     max_a = df_3a['date'].max()
#     min_b = df_3b['date'].min()
    
#     df_3b['date'] = df_3b['date'] + (max_a - min_b) + pd.Timedelta(seconds=0.1)

#     df = pd.concat([df_3a, df_3b], ignore_index=True)

#     # find first index where robot moves
#     threshold = 0.1
#     first_move_index = 0
#     for j in range(len(df['GPS2 (2D) [m/s]'])):
#         if df.loc[j,'GPS2 (2D) [m/s]'] > threshold:
#             first_move_index = j
#             break

#     # remove first_move_index rows from df
#     df = df.iloc[first_move_index:]

#     # find the minimum length of the two dataframes
#     min_length = min(len(df[col]) for col in df.columns)
#     df = df.iloc[:min_length]

#     # Calculate the difference in milliseconds between each row and the first row
#     df['date'] = pd.to_datetime(df['date'])
#     first_date = df.iloc[0]['date']
#     df['milliseconds'] = (df['date'] - first_date).dt.total_seconds() * 1000
#     df['seconds'] = (df['date'] - first_date).dt.total_seconds()

#     # calculate difference in milliseconds between each row and the previous row
#     df['diff_milliseconds'] = df['milliseconds'].diff()
#     df['diff_seconds'] = df['seconds'].diff()

#     # distance from robot to participant and target
#     df['distance'] = df.apply(lambda row: geodesic((row['GPS1 (Lat.) [deg]'], row['GPS1 (Long.) [deg]']), (row['GPS2 (Lat.) [deg]'], row['GPS2 (Long.) [deg]'])).meters, axis=1)
#     df['distance_target'] = df['distance'].apply(lambda x: x if x > 1 else 0)

#     # calculate total distance RMSE
#     distance_rmse = math.sqrt(np.sum((df['distance_target']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#     print(f"distance target RMSE is {distance_rmse}")
#     rmse_results.append(distance_rmse)

#     # Plotting the graph with two lines representing participant's path and robot's path
#     plt.clf()
#     plt.plot(df['GPS1 (Long.) [deg]'], df['GPS1 (Lat.) [deg]'], color='coral', label='participant_path')
#     plt.plot(df['GPS2 (Long.) [deg]'], df['GPS2 (Lat.) [deg]'], color='blue', label='robot_path')
#     plt.xlabel('Long. [deg]', fontsize=8)  # Adjust the font size
#     plt.ylabel('Lat. [deg]', fontsize=8)   # Adjust the font size
#     plt.title('Walking Paths')
#     plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
#     plt.legend()
#     plt.savefig(f'{pre}paths.png')

#     # plotting distance over time
#     line_plot(df, 'seconds', 'distance', 'Distance Between Robot and Participant', 'Time (s)', 'Distance (m)', f'{pre}distance.png')

#     # plotting speeds over time
#     line_plot(df, 'seconds', ['GPS1 (2D) [m/s]', 'GPS2 (2D) [m/s]'], 'Speed Plot', 'Time (s)', 'Speed (m/s)', f'{pre}speeds.png')

#     df['speed_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]'])

#     # plotting speed difference over time
#     line_plot(df, 'seconds', 'speed_difference', 'Speed Difference', 'Time (s)', 'Speed Difference (m/s)', f'{pre}speed_difference.png')

#     df['pace_difference'] = abs(df['GPS1 (2D) [m/s]'] - df['GPS2 (2D) [m/s]']) / (df['GPS1 (2D) [m/s]'] + df['GPS2 (2D) [m/s]'])

#     # printing speed difference RMSE
#     speed_difference_rmse = math.sqrt(np.sum((df['speed_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#     print(f"speed difference RMSE is {speed_difference_rmse}")
#     rmse_results.append(speed_difference_rmse)

#     # plotting pace difference over time
#     line_plot(df, 'seconds', 'pace_difference', 'Pace Asymmetry', 'Time (s)', 'Pace Asymmetry', f'{pre}pace_asymmetry.png')

#     # calculate walking direction of participant and robot
#     df['walking_direction'] = np.arctan2(df['GPS1 (Lat.) [deg]'].diff(), df['GPS1 (Long.) [deg]'].diff()) * 180 / np.pi
#     df['walking_direction_robot'] = np.arctan2(df['GPS2 (Lat.) [deg]'].diff(), df['GPS2 (Long.) [deg]'].diff()) * 180 / np.pi

#     # smoothing walking direction of participant and robot
#     df['walking_direction'] = df['walking_direction'].rolling(10).mean()
#     df['walking_direction_robot'] = df['walking_direction_robot'].rolling(10).mean()

#     # plot walking directions of participant and robot
#     line_plot(df, 'seconds', ['walking_direction', 'walking_direction_robot'], 'Walking Directions', 'Time (s)', 'Walking Direction (deg)', f'{pre}walking_direction.png')

#     # calculate direction difference (angle between participant and robot)
#     df['direction_difference'] = abs(df['walking_direction'] - df['walking_direction_robot'])

#     # if direction_difference > 180, then subtract 180
#     df['direction_difference'] = np.where(df['direction_difference'] > 180, df['direction_difference'] - 180, df['direction_difference'])

#     # printing direction difference RMSE
#     direction_difference_rmse = math.sqrt(np.sum((df['direction_difference']**2)*df['diff_milliseconds'])/df['diff_milliseconds'].sum())
#     print(f"direction difference RMSE is {direction_difference_rmse}")
#     rmse_results.append(direction_difference_rmse)

#     # plot direction difference over time
#     line_plot(df, 'seconds', 'direction_difference', 'Angle of Difference in Walking Direction', 'Time (s)', 'Direction Difference (deg)', f'{pre}direction_difference.png')

#     # save RMSE results to csv
#     cols = ['distance_rmse', 'speed_difference_rmse', 'direction_difference_rmse']
#     rmse_results_df = pd.DataFrame([rmse_results], columns=cols)
#     rmse_results_df.to_csv(f'{pre}rmse_results.csv', index=False)
