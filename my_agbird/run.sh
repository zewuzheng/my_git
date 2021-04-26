#! /bin/bash
python3 my_rule.py 
pidof my_training_bk.py 
while [ $? -ne 0 ]   
do
    echo "Process exits with errors! Restarting!"
    python3 my_rule.py
done
