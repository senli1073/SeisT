
dt=`date +'%Y-%m-%d_%H-%M-%S'`

nohup python -u main_$1.py > logs/log_$dt.log 2>&1 &