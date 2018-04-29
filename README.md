# Interpretable Test 

[![Build Status](https://travis-ci.org/wittawatj/interpretable-test.svg?branch=master)](https://travis-ci.org/wittawatj/interpretable-test)

**17 April 2018**:  We updated the code base to provide support for both Python
3 and Python 2.7. Please contact [Wittawat Jitkrittum](http://wittawat.com) if
you found a bug.

The goal of this project is to learn a set of features to distinguish two given distributions P and Q, as observed through two samples. This task is formulated as a two-sample test problem. The features are chosen so as to maximize the distinguishability of the distributions, by optimizing a lower bound on test power for a statistical test using these features. The result is a parsimonious and interpretable indication
of how and where two distributions differ locally (when the null hypothesis i.e., P=Q is rejected). 

This repository contains a Python implementation of the Mean Embeddings (ME) test, and Smooth Characteristic Function (SCF) test in which features are automatically optimized as described in [our paper](http://arxiv.org/abs/1605.06796)

    Interpretable Distribution Features with Maximum Testing Power
    Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretton
    NIPS, 2016

## How to install?

The package can be installed with the `pip` command.

    pip install git+https://github.com/wittawatj/interpretable-test

Once installed, you should be able to do `import freqopttest` without any error.

## Demo scripts
To get started, check [demo_interpretable_test.ipynb](https://github.com/wittawatj/interpretable-test/blob/master/ipynb/demo_interpretable_test.ipynb) which will guide you through from the beginning. There are many Jupyter notebooks in `ipynb` folder. Be sure to check them if you want to explore more.

## Reproduce experimental results
Each experiment is defined in its own Python file with a name starting with `exXX` where `XX` is a number. All the experiment files are in `freqopttest/ex` folder. Each file is runnable with a command line argument. For example in `ex1_power_vs_n.py`, we aim to check the test power of each testing algorithm as a function of the sample size `n`. The script `ex1_power_vs_n.py` takes a dataset name as its argument. See `run_ex1.sh` which is a standalone Bash script on how to execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/wittawatj/independent-jobs) package to parallelize our experiments over a [Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you just need to use our developed two-sample tests). For example, for `ex1_power_vs_n.py`, a job is created for each combination of `(dataset, algorithm, n, trial)`. If you do not use Slurm, you can change the line 

    engine = SlurmComputationEngine(batch_parameters)

to 

    engine = SerialComputationEngine()

which will instruct the computation engine to just use a normal for-loop on a single machine (will take a lot of time). Other computation engines that you use might be supported. See  [independent-jobs's repository page](https://github.com/wittawatj/independent-jobs). For real-data experiments, all the preprocessed data are included in `freqopttest/data/` as Pickle files. An experiment script will create a lot of results saved as Pickle files in `freqopttest/result/exXX/` where `XX` is the experiment number. To plot these results, see the experiment's corresponding Jupyter notebook in the `ipynb/` folder. For example, for `ex1_power_vs_n.py` see `ipynb/ex1_results.ipynb` to plot the results.

## Preprocessed NIPS text collection
We will add a link to the proprocessed collection of NIPS papers from 1988 to 2015 that we used in the paper soon. All the scripts used will also be added. Stay tuned.

## License
[MIT license](https://github.com/wittawatj/interpretable-test/blob/master/LICENSE).

If you have questions or comments about anything regarding this work, please do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com).
