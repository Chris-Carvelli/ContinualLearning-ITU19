##### Source code for Thesis Project
# Techniques for Continual Learning
### Hypernetworks and Neural Turing Machines as a stepping stone towards meta-learning agents

This project contains an implementation of a Genetic Algorithm used for 
artificial neural network and was used to run several experiments using a different models,
including a [Evolvable Neural Turing Machine](http://sebastianrisi.com/wp-content/uploads/greve_ram15.pdf) 
and a [Hypernetwork](https://arxiv.org/abs/1609.09106).


## Getting started
Clone the repository and install the packages from requirement.txt 

In order to 
Use the main.py with several commandline options (use the --help option if needed). 
Below are some examples:

#### Training
Train a Hypernetwork on the DistShift environment (see config file for details)\
`python main.py run --config_name DistShift.cfg --config_folder config_files/minigrid/HyperNN`

Train with 5 experimental runs. Removing `-pe` option finishes the first run before starting on the next run.\
`python main.py run --config_name DistShift.cfg --config_folder config_files/minigrid/HyperNN --multi_session 5 --pe`

A session file name after the config file is automatically created. 
If session file with the same name already exists, it attempts to load and continue the that session. 

#### Plotting
Use the plot command to get a plot of max, median, and mean score during evolution.
The script will guide you to find the session file normally located in a folder called Experiments\
`python main.py plot`

The plot command has a multitude of options and plot more than one session at time.
Use the `python main.py plot --help` command for a description of options.

#### Render
This command shows loads a models and renders them using the OpenAI's `env.render()` function.
This allows you to see how the againt actually behaves in the environment. 
Like the plot command it will guide you to find the right session file, 
but also has a multitude of options. Use `--help` options for more info.\
`python main.py plot`


