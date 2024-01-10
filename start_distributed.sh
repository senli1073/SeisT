

export OMP_NUM_THREADS='1'

dt=`date +'%Y-%m-%d_%H-%M-%S'`

nohup torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    main.py \
    > logs/log_"$dt"_dist.log 2>&1 &

