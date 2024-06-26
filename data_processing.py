from utils import *

############################################################################################################
""" Parameters """

only_407 = False
# set_debug(True)
create_folders_instead = True

basic_plots_and_rmse = False
advanced_plots = True
advanced_plots_only_person = False

verbose_participant = True
verbose_read_csv_files = True
verbose_basic_rmse = False
verbose_compute_turning = False
verbose_plot_turning = True
verbose_plot_main_map_with_turns = False
verbose_plot_all_turns = True


encode_walking_direction_params = {'direction_in_cluster_thres': 1, 'cluster_leak_thres': 1, 'starting_leak_allowance': -5}
compute_basic_turnings_params = {'inter_cluster_is_turn_thres': 50, 'turning_window_size': 12, 'start_end_buffer': 50}
remove_non_desicive_turns_params = {'avg_radius_pre_post_turn_limit': 40, 'ignore_turn_pre_post_avgs_diff_thres': 50}
plot_main_map_with_turns_params = {'black_radius': 2, 'black_mod': 50, 'blue_radius': 2, 'start_thres': 20, 
                                   'turnings': True, 'green_start': False, 'red_end': True, 
                                   'black_marks': True, 'blue_marks': True, 'grey_marks': True}
plot_all_turns_plot_options = {'single_turn_paths': True, 
                               'distance': True, 'speed_difference': True, 'walking_direction_difference': True, 
                               'speeds': True, 'walking_directions': True}
    

############################################################################################################
""" Main Code """

collect_all_turns = {}
clean_DS_store_files()
normal_participants_list = [407,408]+[i for i in range(2101,2112) if i != 2110]

for participant_id in normal_participants_list:
    collect_all_turns[f"{participant_id}"] = {}
    if verbose_participant:
        print(f'participant {participant_id}')
    
    results_folder, turns_folder = create_base_folders(participant_id, instead=create_folders_instead if participant_id==407 else False)
    
    for path_num in range(0, 3):
        
        parent_dir = f'./data/{participant_id}/'
        df_person, df_robot = read_csv_files(parent_dir, path_num, verbose=verbose_read_csv_files)
        df = preprocess_join(df_person, df_robot)
        df = add_cols(df)
        pre = results_folder + f'{str(participant_id)}_{str(path_num+1)}_'
        
        if basic_plots_and_rmse:
            basic_plots(df, pre)
            rmse_results = basic_rmse(df, pre, verbose=verbose_basic_rmse)

        df = fix_walking_direction(df)

        X = get_clean_walking_direction_np(df, 'walking_direction') 
        X_robot = get_clean_walking_direction_np(df, 'walking_direction_robot')
        
        walking_direction_encoded = encode_walking_direction(X, **encode_walking_direction_params)
        walking_direction_encoded_robot = encode_walking_direction(X_robot, **encode_walking_direction_params)
        
        df['walking_direction_encoded'] = walking_direction_encoded
        df['walking_direction_encoded_robot'] = walking_direction_encoded_robot
        
        if verbose_compute_turning:
            print("Person:")
        is_turning = compute_basic_turnings(df, walking_direction_encoded, 
                                            verbose=verbose_compute_turning, **compute_basic_turnings_params)
        is_turning = remove_non_decisive_turns(df, walking_direction_encoded, is_turning, 
                                               verbose=verbose_compute_turning, **remove_non_desicive_turns_params)
        df['is_turning'] = is_turning
        lmrs = get_lmrs(is_turning, verbose=verbose_compute_turning)
        
        if verbose_compute_turning:
            print("Robot:")
        is_turning_robot = compute_basic_turnings(df, walking_direction_encoded_robot, 
                                                  verbose=verbose_compute_turning, **compute_basic_turnings_params)
        is_turning_robot = remove_non_decisive_turns(df, walking_direction_encoded_robot, is_turning_robot, 
                                                     verbose=verbose_compute_turning, **remove_non_desicive_turns_params)
        df['is_turning_robot'] = is_turning_robot
        lmrs_robot = get_lmrs(is_turning_robot, verbose=verbose_compute_turning)
        
        if advanced_plots:
            plot_advanced(df, pre, advanced_plots_only_person)
        
        df = discretize_turnings(df)
        
        plot_main_map_with_turns(df, turns_folder + f'{str(participant_id)}_{str(path_num+1)}_', lmrs, lmrs_robot, 
                                 verbose=verbose_plot_main_map_with_turns, **plot_main_map_with_turns_params)

        stats_person_turns = plot_all_turns(df, turns_folder, path_num, lmrs, person=True, verbose=verbose_plot_all_turns, save_to_csv=True, plot_options=plot_all_turns_plot_options)
        stats_robot_turns = plot_all_turns(df, turns_folder, path_num, lmrs_robot, person=False, verbose=verbose_plot_all_turns, save_to_csv=True, plot_options=plot_all_turns_plot_options)
        collect_all_turns[f"{participant_id}"][f"{path_num}"] = {'person': stats_person_turns, 'robot': stats_robot_turns}
    if only_407:
        break

new_collect = []
for participant in collect_all_turns:
    for path in collect_all_turns[participant]:
        for person_robot in collect_all_turns[participant][path]:
            try:
                for i in range(len(eval(str(collect_all_turns[participant][path][person_robot]['walking_direction_lag'])))):
                        item_dict = {"participant_id": participant, "path_num": int(path)+1, "person_robot": person_robot, "turn_num": i+1}
                        for stat in collect_all_turns[participant][path][person_robot]:
                            item_dict[f"{stat}"] = eval(str(collect_all_turns[participant][path][person_robot][stat]))[i]
                        new_collect.append(item_dict)
            except:
                continue
df = pd.DataFrame(new_collect)
df.to_csv("all_turns.csv")