#!/bin/bash

ulimit -n 65536
export RAY_HEAD_SERVICE_PORT=6379
export NODE_IP_ADDRESS=192.168.1.3
export CUDA_VISIBLE_DEVICES="7"

python combine_py_files.py __main__.py main_selfplay_combined.py | tee -a log.txt

export RAY_HEAD_SERVICE_HOST=`kubectl -n rayselfplay get svc | grep ray-head-selfplay | awk -v N=3 '{print $N}'`

restart_cluster=$1

for i in `seq 1 10000`; do
	if [ $restart_cluster -eq 1 ]; then
		echo "Restarting k8s cluster" | tee -a log.txt
		ray stop
		kubectl delete -f rayselfplay-cluster.yaml 2>&1 | tee -a log.txt
		kubectl apply -f rayselfplay-cluster.yaml 2>&1 | tee -a log.txt
		sleep 20
		while [[ `kubectl -n rayselfplay get pods | grep Running | wc -l` -lt "20" ]]; do
			sleep 2
			echo `kubectl -n rayselfplay get pods | grep Running | wc -l` "/ 20"
		done
		sleep 2
	   	export RAY_HEAD_SERVICE_HOST=`kubectl -n rayselfplay get svc | grep ray-head-selfplay | awk -v N=3 '{print $N}'`
		ray start --address=$RAY_HEAD_SERVICE_HOST:$RAY_HEAD_SERVICE_PORT --node-ip-address=$NODE_IP_ADDRESS 2>&1 | tee -a log.txt
	fi
	echo "Starting python script" | tee -a log.txt
        python -u main_selfplay_combined.py --seed 0 2>&1 | tee -a log.txt
	echo "Ended python script" | tee -a log.txt
	sleep 5
	#if [ $(($i % 5)) -eq 0 ]; then
	#	restart_cluster=1
	#else
	#	restart_cluster=0
	#fi
	restart_cluster=1
done
