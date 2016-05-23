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
4. Check that indeed the package is in the search path by openning a new Python shell, and issuing `import freqopttest`. If there is no import error, the installation is completed.  


## Demo script
See ... 


* To reproduce experimental results,
  [independent-jobs](https://github.com/karlnapf/independent-jobs) package must
be in the PYTHONPATH. 


