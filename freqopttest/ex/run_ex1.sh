#!/bin/bash 

screen -AdmS ex1_fotest -t tab0 bash 

#python ex1_power_vs_n.py sg_d50
#python ex1_power_vs_n.py gmd_d100 
#python ex1_power_vs_n.py SSBlobs
#python ex1_power_vs_n.py gvd_d50

# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex1_fotest -X screen -t tab1 bash -lic "python ex1_power_vs_n.py sg_d50"
screen -S ex1_fotest -X screen -t tab2 bash -lic "python ex1_power_vs_n.py gmd_d100"
screen -S ex1_fotest -X screen -t tab3 bash -lic "python ex1_power_vs_n.py SSBlobs"
screen -S ex1_fotest -X screen -t tab4 bash -lic "python ex1_power_vs_n.py gvd_d50"

