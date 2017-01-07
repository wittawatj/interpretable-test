#!/bin/bash 

screen -AdmS ex2_fotest -t tab0 bash 

##python ex2_vary_d.py sg_low
#python ex2_vary_d.py gvd
#python ex2_vary_d.py gmd
#python ex2_vary_d.py sg

# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex2_fotest -X screen -t tab1 bash -lic "python ex2_vary_d.py gvd"
screen -S ex2_fotest -X screen -t tab2 bash -lic "python ex2_vary_d.py gmd"
screen -S ex2_fotest -X screen -t tab3 bash -lic "python ex2_vary_d.py sg"

