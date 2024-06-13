from utils import *

############################################################################################################
""" Parameters """

DEBUG = False
only_407 = True
create_folders_instead = True

basic_plots_and_rmse = False
advanced_plots = True
advanced_plots_only_person = False

verbose_participant = False
verbose_read_csv_files = False
verbose_basic_rmse = False
verbose_compute_turning = False
verbose_plot_turning = False
verbose_plot_main_map_with_turns = False


encode_walking_direction_params = {'direction_in_cluster_thres': 1, 'cluster_leak_thres': 1, 'starting_leak_allowance': -5}
compute_turnings_legacy_params = {'inter_cluster_is_turn_thres': 50, 'avg_radius_pre_post_turn_limit': 40, 'turning_window_size': 20, 'start_end_buffer': 50}
plot_main_map_with_turns_params = {'black_radius': 2, 'black_mod': 50, 'blue_radius': 2, 'start_thres': 20, 
                                   'turnings': True, 'green_start': False, 'red_end': True, 
                                   'black_marks': True, 'blue_marks': True, 'grey_marks': True}

############################################################################################################
""" Main Code """

clean_DS_store_files()
normal_participants_list = [407,408]+[i for i in range(2101,2112) if i != 2110]

for participant_id in normal_participants_list:
    if verbose_participant:
        print(f'participant {participant_id}')
    
    results_folder, turns_folder = create_base_folders(participant_id, instead=create_folders_instead)
    
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
        is_turning = compute_turnings_legacy(df, walking_direction_encoded, verbose=verbose_compute_turning, **compute_turnings_legacy_params)
        df['is_turning'] = is_turning
        lmrs = get_lmrs(is_turning, verbose=verbose_compute_turning)
        
        if verbose_compute_turning:
            print("Robot:")
        is_turning_robot = compute_turnings_legacy(df, walking_direction_encoded_robot, verbose=verbose_compute_turning, **compute_turnings_legacy_params)
        df['is_turning_robot'] = is_turning_robot
        lmrs_robot = get_lmrs(is_turning_robot, verbose=verbose_compute_turning)
        
        if advanced_plots:
            plot_advanced(df, pre, advanced_plots_only_person)
        
        df = discretize_turnings(df)
        
        plot_main_map_with_turns(df, turns_folder + f'{str(participant_id)}_{str(path_num+1)}_', lmrs, lmrs_robot, verbose=verbose_plot_main_map_with_turns, **plot_main_map_with_turns_params)

        plot_all_turns(df, turns_folder, path_num, lmrs, person=True, verbose=verbose_plot_turning)
        plot_all_turns(df, turns_folder, path_num, lmrs_robot, person=False, verbose=verbose_plot_turning)

    if only_407:
        break