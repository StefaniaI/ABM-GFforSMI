# This file creates the config files for running the experiments
import os
import scipy.stats as st
import numpy as np


def read_seeds(file_name='seeds_10000.txt'):
    'Reads the file with the 10000 random seeds'

    f = open(file_name, 'r')
    lines = f.read().splitlines()
    lines = [int(l) for l in lines]
    f.close()

    return lines


def generate_config_csvs(base_config, changing, var_lists, path='Simulation_results',
                         no_folders=1, start_config_no=-1, random_seed_generate=False):
    '''Creates the .csv files given the the fix parameters and the cahnging parameters.
    base_config = dictionary with the fixed parameters {parameter_name: parameter_val, ...}
    changing = dictionary with the changing parameters {[p1, p2]:[[v1, v2], [v1.0, v2.0]], ...}
    random_seed_generate = instead of taking from the config the random seed generate it now at random'''

    if random_seed_generate:
        np.random.seed(97)

    def write_parameters(dict_param, no_comb):
        ''' Writes a dictionary of parameters to a file indexed by no_comb.
        '''
        current_path = os.path.join(path, 'config' + str(no_comb) + '.csv')
        with open(current_path, 'w') as f:
            for key in dict_param.keys():
                f.write("%s,%s\n" % (key, dict_param[key]))

    config = {}
    no = start_config_no

    # put the base parameters into config
    for var in base_config:
        config[var] = base_config[var]

    # generate all combinations for the changing parameters
    choice_changing = [0 for i in var_lists]
    constructed_all = False

    while not constructed_all:
        # set all variables according to the current choice of parameter combinations
        for var_list_order_no in changing:
            var_list = var_lists[var_list_order_no]
            for var_pos in range(len(var_list)):
                var = var_list[var_pos]
                alternative_no = choice_changing[var_list_order_no]
                config[var] = changing[var_list_order_no][alternative_no][var_pos]

                # generate random seed if this is the current variable
                if random_seed_generate and var == 'random_seed':
                    config[var] = np.random.randint(1000000)

        # save the current config file
        no += 1
        write_parameters(config, no)

        # iterate twards the next choice_changing
        constructed_all = True
        for var_list_order_no in range(len(var_lists)):
            no_options = len(changing[var_list_order_no])

            # if we can increase
            if choice_changing[var_list_order_no] < (no_options - 1):
                # increase the choce option
                choice_changing[var_list_order_no] += 1
                # the ones before are reseted to 0
                for j in range(var_list_order_no):
                    choice_changing[j] = 0
                # we didn't construct all - there is a new choice
                constructed_all = False
                break

    no += 1

    # create no_folders folders with the names of the generated files
    for i in range(no_folders):
        current_path = os.path.join(path, "file_names" + str(i + 1) + ".txt")
        f = open(current_path, mode='w')
        for j in range(start_config_no + 1, no):
            if j % no_folders == i:
                print_path = os.path.join(path, 'config' + str(j) + '.csv\n')
                f.writelines((print_path))

    return no


def configs_ve_test(no_seeds=1):
    '''This function creates the files that experiment with different random seeds for the setup:
    - multidimensional with one competing attribute taste
    - alpha = 0
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '[1]'
    config['cumulative_weights'] = -1
    config['kind_attributes'] = "['c']"
    config['dict_cov'] = '"{(0, 0): 0}"'
    config['matching_bound'] = 0
    config['prob_cumulative_weights'] = '[0.5, 0.5]'

    config['num_steps'] = 365

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], ['random_seed']]

    changing[0] = [['AntiPA'], ['PA-AntiPA']]

    seeds = read_seeds()
    changing[1] = [[s] for s in seeds[:no_seeds]]

    return config, changing, changing_var_lists


def configs_ve_1a(no_seeds=1, start_seed=0):
    '''This function creates the files that experiment with different random seeds for
    RQ 1a: How does the size of the minority impact the fiarness of CCs?
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '"[1, 1]"'
    config['kind_attributes'] = '''"['c', 'm']"'''
    config['dict_cov'] = '"{(0, 0): 0, (0, 1):0, (1, 1):0}"'
    config['cumulative_weights'] = '[-1]'
    config['prob_cumulative_weights'] = '[1]'

    config['num_steps'] = 0

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], [
        '%_groupA', 'matching_bound'], ['random_seed']]

    changing[0] = [['PA'], ['UR'], ['ExtremePA']]
    changing[1] = [[0, st.norm.ppf(0.0000000001)], [0.1, st.norm.ppf(0.1)], [0.2, st.norm.ppf(
        0.2)], [0.3, st.norm.ppf(0.3)], [0.4, st.norm.ppf(0.4)], [0.5, st.norm.ppf(0.5)]]

    seeds = read_seeds()
    changing[2] = [[s] for s in seeds[start_seed:(start_seed + no_seeds)]]

    return config, changing, changing_var_lists


def configs_ve_1b(no_seeds=1, start_seed=0):
    '''This function creates the files that experiment with different random seeds for
    RQ 1b: How does the level of bias in the population impact the fiarness of CCs?
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '"[1, 1]"'
    config['kind_attributes'] = '''"['c', 'm']"'''
    config['dict_cov'] = '"{(0, 0): 0, (0, 1):0, (1, 1):0}"'
    config['prob_cumulative_weights'] = '[1]'

    config['num_steps'] = 0

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], [
        '%_groupA', 'matching_bound'], ['cumulative_weights'], ['random_seed']]

    changing[0] = [['PA'], ['UR'], ['ExtremePA']]
    changing[1] = [[0.25, st.norm.ppf(0.25)], [0.5, st.norm.ppf(0.5)]]
    changing[2] = [['"[[1, 0]]"'], ['"[[0.75, 0.25]]"'],
                   ['"[[0.5, 0.5]]"'], ['"[[0.25, 0.75]]"'], ['"[[0.01, 0.99]]"']]

    seeds = read_seeds()
    changing[3] = [[s] for s in seeds[start_seed:(start_seed + no_seeds)]]

    return config, changing, changing_var_lists


def configs_ve_1c(no_seeds=1, start_seed=0):
    '''This function creates the files that experiment with different random seeds for
    RQ 1c: How does the percentage of bias in the population impact the fiarness of CCs?
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '"[1, 1]"'
    config['kind_attributes'] = '''"['c', 'm']"'''
    config['dict_cov'] = '"{(0, 0): 0, (0, 1):0, (1, 1):0}"'
    config['cumulative_weights'] = '"[[1, 0], [0.5, 0.5]]"'

    config['num_steps'] = 0

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], [
        '%_groupA', 'matching_bound'], ['prob_cumulative_weights'], ['random_seed']]

    changing[0] = [['PA'], ['UR'], ['ExtremePA']]
    changing[1] = [[0.25, st.norm.ppf(0.25)], [0.5, st.norm.ppf(0.5)]]
    changing[2] = [['"[1, 0]"'], ['"[0.75, 0.25]"'],
                   ['"[0.5, 0.5]"'], ['"[0.25, 0.75]"'], ['"[0, 1]"']]

    seeds = read_seeds()
    changing[3] = [[s] for s in seeds[start_seed:(start_seed + no_seeds)]]

    return config, changing, changing_var_lists


def configs_ve_2(no_seeds=1, start_seed=0):
    '''This function creates the files that experiment with different random seeds for
    RQ 2: How does the level of biase in RS impact the fiarness of CCs?
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '"[1, 1]"'
    config['kind_attributes'] = '''"['c', 'm']"'''
    config['dict_cov'] = '"{(0, 0): 0, (0, 1):0, (1, 1):0}"'
    config['prob_cumulative_weights'] = '[1]'

    config['num_steps'] = 0

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], ['level_bias_RS'], [
        '%_groupA', 'matching_bound'], ['cumulative_weights'], ['random_seed']]

    changing[0] = [['biased_PA'], ['biased_UR'], ['biased_ExtremePA']]
    changing[1] = [[0], [0.25], [0.5], [0.75]]
    changing[2] = [[0.25, st.norm.ppf(0.25)], [0.5, st.norm.ppf(0.5)]]
    # changing[2] = [[0.25, st.norm.ppf(0.25)]]
    # changing[3] = [['"[[1, 0]]"'], ['"[-1]"']]
    changing[3] = [['"[[1, 0]]"']]

    seeds = read_seeds()
    changing[4] = [[s] for s in seeds[start_seed:(start_seed + no_seeds)]]

    return config, changing, changing_var_lists


def configs_ve_3(no_seeds=1, start_seed=0):
    '''This function creates the VE for RQ3a and RQ3b.
    RQ3a: What is more effective reducing the level of bias in population or RS?
    RQ3b: Can we overcome the population bias by a reversed-biased RS?
    We vary:
    - bias_level (for biased users): 0.5, 0.75
    - prob_cumulative_weights: 0.75, 0.5, 0.25 (i.e. % unbiased users)
    - level_bias_RS: 0.5, 0.25, 0, -0.25, -0.5
    '''

    # start by setting the parameters that will remain fixed
    config = {}

    config['num_users'] = 1000
    config['num_items'] = 50

    config['type_attributes'] = 'multidimensional'
    config['num_attributes'] = '"[1, 1]"'
    config['kind_attributes'] = '''"['c', 'm']"'''
    config['dict_cov'] = '"{(0, 0): 0, (0, 1):0, (1, 1):0}"'
    config['%_groupA'] = 0.25
    config['matching_bound'] = st.norm.ppf(0.25)

    config['num_steps'] = 0

    # set the parameters we varry
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    changing_var_lists = [['rs_model'], [
        'cumulative_weights'], ['prob_cumulative_weights'], ['level_bias_RS'], ['random_seed']]

    changing[0] = [['biased_PA'], ['biased_UR'], ['biased_ExtremePA']]
    # trebuia sa fie: changing[1] = [['"[[1, 0], [0.5, 0.5]]"'], ['"[[1, 0], [0.25, 0.75]]"'], ['"[[1, 0], [0.01, 0.99]]"']]
    changing[1] = [['"[[1, 0], [0.5, 0.5]]"'], [
        '"[[1, 0], [0.25, 0.75]]"'], ['"[[1, 0], [0.01, 0.99]]"']]
    changing[2] = [['"[1, 0]"'], ['"[0.75, 0.25]"'],
                   ['"[0.5, 0.5]"'], ['"[0.25, 0.75]"'], ['"[0, 1]"']]
    changing[3] = [[0], [0.25], [0.5], [-0.25], [-0.5]]

    seeds = read_seeds()
    changing[4] = [[s] for s in seeds[start_seed:(start_seed + no_seeds)]]

    return config, changing, changing_var_lists
