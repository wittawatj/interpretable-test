#!/bin/bash 

#python ex4_text.py bayes_bayes_d2000_rnoun
#python ex4_text.py bayes_deep_d2000_rnoun
##python ex4_text.py bayes_neuro_d300_rnoun
#python ex4_text.py bayes_neuro_d2000_rnoun
##python ex4_text.py bayes_neuro_d800_rverb
##python ex4_text.py deep_learning_d1000_rnoun
#python ex4_text.py deep_learning_d2000_rnoun
#python ex4_text.py bayes_learning_d2000_rnoun
#python ex4_text.py neuro_learning_d2000_rnoun
##python ex4_text.py bayes_deep_d1000_rnoun
##python ex4_text.py deep_neuro_d2000_rnoun

screen -AdmS ex4_fotest -t tab0 bash 

# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py bayes_bayes_d2000_rnoun"
screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py bayes_deep_d2000_rnoun"
screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py bayes_neuro_d2000_rnoun"
screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py deep_learning_d2000_rnoun"
screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py bayes_learning_d2000_rnoun"
screen -S ex4_fotest -X screen -t tab1 bash -lic "python ex4_text.py neuro_learning_d2000_rnoun"

