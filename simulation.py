import time
import json
import numpy as np
import csv
import re
import os
import ast

import model


class Simulation:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = self.read_config()
        self.results = []

    def get_sim_no(self):
        '''Gets the number of the simulation from the file_name'''

        numbers = re.findall(r'\d+', self.file_name)

        if len(numbers) == 0:
            return ''

        return numbers[0]

    def get_path(self):
        '''If the file_name also has a path to another folder, keeps that separate'''

        file_name_components = self.file_name.split('/')

        if len(file_name_components) == 1:
            return ''

        return file_name_components[:-1]

    def read_config(self):
        ''' Reads the file with parameters, given its name.'''

        infile = open(self.file_name, mode='r')
        reader = csv.reader(infile)
        config = {rows[0]: rows[1] for rows in reader}

        # transforms non-string parameters to correct type
        str_param_names = ['Variable name', 'type_attributes', 'distribution',
                           'rs_model']
        for name in config:
            if name not in str_param_names:
                config[name] = ast.literal_eval(config[name])

        self.config = config

        return config

    def return_results(self):
        '''This function returns the results of a simulation without saving it'''

        no = self.get_sim_no()
        path = self.get_path()
        f = open(os.path.join(*path, 'run' + str(no) + '.json'), 'r')
        data = f.read()
        f.close()

        return json.loads(data)

    def read(self, all=True):
        '''This takes the results and configuration for a simulation that was run before'''

        # read the config file
        self.read_config()

        # load the results into the simulation object
        if all:
            no = self.get_sim_no()
            path = self.get_path()
            f = open(os.path.join(*path, 'run' + str(no) + '.json'), 'r')
            data = f.read()
            f.close()

            self.results = {**json.loads(data), **self.config}
            # self.results = pd.read_csv(os.path.join(*path, 'run' + str(no) + '.csv'))

    def simulate(self, read_config=True):
        '''Runs a simulation, for the parameters in the config file.
        This updates the object ranther than returning the results.
        '''

        # Load the parameters from the file "config.csv"
        no_file = self.get_sim_no()

        # variables to track during simulation
        data = {'num_file_sim': no_file}

        # set the random seed
        np.random.seed(self.config['random_seed'])

        # create the platform
        p = model.Platform(self.config)

        # iterate the platform either num_steps or until convergence
        num_iterations = self.config['num_steps']
        if num_iterations:
            for i in range(num_iterations):
                num_iterations += 1
                p.iterate()
        else:
            while not p.check_convergence():
                num_iterations += 1
                p.iterate()
                p.update_searching_users()

        # record statistics after the runs
        data['num_followers'] = p.network.num_followers.tolist()
        data['num_followees'] = p.network.num_followees.tolist()
        data['timesteps'] = num_iterations
        data['G'] = p.network.G.tolist()

        # metrics for user fairness
        data['num_timestep_users_found_best'] = p.users_found_best
        data['users_pos_rec'] = p.users_rec_pos
        data['recommended_maching_CC'] = p.rec_same_maching
        # metric for CC fairness
        data['num_users_recommended_CC'] = p.num_users_rec_CC

        # record the borda score (--> order of items under different preferences)
        data['borda_original'] = list(p.get_borda_scores())
        data['borda_power'] = list(p.get_borda_scores(rule='power'))
        data['quality'] = list(p.get_competing_scores())

        # record the community of each content creator (maching attributes) - ToDo:remove
        data['maching'] = [c.maching_attr for c in p.CCs]
        # record whether or not users and CCs are protected
        data['is_user_protected'] = [u.protected for u in p.users]
        data['is_CC_protected'] = [c.protected for c in p.CCs]

        # record the results
        self.results = data


def run_sim(file_name="Simulation_results/config.csv"):
    '''Runs the simulation for one config file. The resulting dataframe is saved into a .csv
    path = by default it saves the results into a separate folder;
           define path as the empty string if you want it to be in the saved in the same folder'''

    # Run a simulation with the given parameters
    sim = Simulation(file_name)
    sim.simulate()
    path = "Simulation_results"

    # Find the config file number
    no = sim.get_sim_no()

    # print(sim.results['attributes'])
    # print(sim.results['strategy'])
    # print(sim.results['utility'])

    # Save the results into the respective folder
    save_path = os.path.join(path, 'run' + str(no) + '.json')
    f = open(save_path, 'w')
    # print(sim.results)
    json.dump(sim.results, f)
    f.close()
    # sim.results.to_csv(save_path)


def run_sims(file_name_configs='Simulation_results/file_names1.txt'):
    '''Runs multiple configXXXXX.csv files.
    file_name_configs = a folder with all the names of the config files that need to be runned.
    '''

    file_name_configs = os.path.join(*file_name_configs.split('/'))
    file_names = open(file_name_configs, mode='r').readlines()
    for file_name in file_names:
        file_name = file_name.rstrip("\n\r")
        print(file_name)
        run_sim(file_name)
