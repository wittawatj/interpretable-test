# Interpretable Test 

The goal of this project is to learn a set of features to distinguish two given distributions P and Q, as observed through two samples. The features are chosen so as to maximize the distinguishability of the distributions, by optimizing a lower bound on test power for a statistical test using these features. The result is a parsimonious and interpretable indication
of how and where two distributions differ locally (when the null hypothesis i.e., P=Q is rejected). This repository contains a Python implementation of the Mean Embeddings (ME) test, and Smooth Characteristic Function (SCF) test with features automatically optimized as described in 

    Interpretable Distribution Features with Maximum Testing Power
    Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretton
    arXiv. May, 2016.

## How to install?
1. Make sure that you have a complete [Scipy stack](https://www.scipy.org/stackspec.html) installed. One way to guarantee this is to install it using [Anaconda with Python 2.7](https://www.continuum.io/downloads), which is also the environment we used to develop this package.
2. Clone or download this repository. You will get the `interpretable-test` folder.
3. Add the path to the folder to Python's seacrh path i.e., to `PYTHONPATH` global variable. See, for instance, [this page on stackoverflow](http://stackoverflow.com/questions/11960602/how-to-add-something-to-pythonpath) on how to do this in Linux. See [here](http://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7) for Windows. 
4. Check that indeed the package is in the search path by openning a new Python shell, and issuing `import freqopttest` (`freqopttest` is the name of our Python package). If there is no import error, the installation is completed.  

## Demo scripts
To get started, check ... which will guide you through from the beginning. There are many Jupyter notebooks in `ipynb` folder. Be sure to check them if you want to explore more.

## Reproduce experimental results,
Each experiment is defined in its own Python file with a name starting with `exXX` where `XX` is a number. All the experiment files are in `freqopttest/ex` folder. Each file is runnable with a command line argument. For example in `ex1_power_vs_n.py`, we aim to check the test power of each testing algorithm as a function of the sample size `n`. The script `ex1_power_vs_n.py` takes a dataset name as its argument. See `run_ex1.sh` which is a standalone Bash script on how to call execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/karlnapf/independent-jobs) package to parallelize our experiments over a [Slurm](http://slurm.schedmd.com/) cluster. For example, for `ex1_power_vs_n.py`, a job is create for each combination of (dataset, algorithm, n, trial). If you do not use Slurm, you can change the line 

    engine = SlurmComputationEngine(batch_parameters)

to 

    #engine = SerialComputationEngine()

which will just use a normal for-loop on a single machine. Other computation engines that you use might be supported. See  [independent-jobs's repository page](https://github.com/karlnapf/independent-jobs). For real-data experiments, all the preprocessed data are included in `freqopttest/data/` as Pickle files. An experiment script will create a lot of results saved as Pickle files in `freqopttest/result/exXX/` where `XX` is the experiment number. To plot these results, see the experiment's corresponding Jupyter notebook in the `ipynb/` folder. For example, for `ex1_power_vs_n.py` see `ipynb/ex1_results.ipynb` to plot the results.




