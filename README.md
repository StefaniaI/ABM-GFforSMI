# Group Fairness for Social Media Influencers

This repository contains the code for the simulation presented in "Group Fairness for Content Creators: the Role of Human and Algorithmic Biases under Popularity-based Recommendations" publish within the Proceedings of the 17th ACM Conference on Recommender Systems (RecSys 2023). We do not include the data resulted from the simulation as the respective file is extremely large (almost 180GB). Regenerating the data takes about three days when running it on three threads (see Computing Infrastructure for more details). The code extends the one we developed previously for understanding individual fairness for social media influencers (https://github.com/StefaniaI/ABM-IFforSMI), which accompanies the paper "The role of luck in the success of social media influencers" (Applied Network Science, 2023). This repository also includes a file with supplementary materials for the aforementioned  paper. The pdf contains diagrams which graphically ilustrate the model and additional sensitivity analysis.

## Overview of the python modules
The classes for the different components of the system (attributes, users, content creators, recommender system, platform) are in model.py. The functions which run simulations and record the fairness metrics can be found in simulation.py. Experimental_setup.py generates the configuration files for the simulations. To see an example of a config file together with explanations on the different parameters please check config.csv. Note that it has more flexibility, as you can, e.g., increase the number of matching, competing and protected attributes. Finally, Data analysis.ipynb Jupyter Notebook contains the code for analysing the data. 

## Run simulation and plotting the results

### Run the simulation
The following commands generate the config files together with four *.tex files which divide the simulations between four threads. The no_folders parameter dictates the number of threads to be used in the process (for speed-ups use larger numbers, depending on the computing infrastructure). Importantly, you need to create a folder named 'Simulation_results' before running this script (the *.confg files will be saved there).

```python
import simulation as sim
import experimental_setup as exp

a, b, c = exp.configs_ve_3(500)
exp.generate_config_csvs(a, b, c, start_config_no=-1, no_folders = 4)
```

To run the config files from thread one, run the following command line in python. You can repeat the process for the remaining threads by running the command with 'Simulation_results/file_namesX.txt' for each X between 1 and no_folders. If you want one thread alone you can run the lines above with no_folders = 1. Please make sure you have at least 180GB free before starting simulations.

```python
import simulation as sim

sim.run_sims(file_name_configs='Simulation_results/file_names1.txt')
```

Combined, the scripts above will generate all the required data for the Jupyter Notebook doing the data analysis in the paper. More precisely you will have one file for each config file (i.e., runX.json contains the reuslts corresponding to configX.csv).

### Plot the results
The Jupyter Notebook data_analysis.ipynb contains the code used to generate the plots in the paper. The sections in the Notebook are organized by the messages shared in the paper. Additional ploting functions are in data_analysis.py.

## Computing Infrastructure
To run the simulations we used Python 3.8.1 on a macOS with Ventura 13.0.1, RAM: 16GB, CPU: Intel Iris Plus Graphics 655 1536 MB.
