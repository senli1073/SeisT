
dt=`date +'%Y-%m-%d_%H-%M-%S'`

nohup python -u main.py > logs/log_$dt.log 2>&1 &