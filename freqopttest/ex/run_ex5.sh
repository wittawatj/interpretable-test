#!/bin/bash 

##python ex5_face.py S48_HANESU_AFANDI
#python ex5_face.py crop48_h0
#python ex5_face.py crop48_HANESU_AFANDI

#!/bin/bash 

screen -AdmS ex5_fotest -t tab0 bash 

# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex5_fotest -X screen -t tab1 bash -lic "python ex5_face.py crop48_h0"
screen -S ex5_fotest -X screen -t tab2 bash -lic "python ex5_face.py crop48_HANESU_AFANDI"

