#!/bin/bash

ulimit -n 65536
export RAY_HEAD_SERVICE_PORT=6379
export NODE_IP_ADDRESS=192.168.1.3
export CUDA_VISIBLE_DEVICES="7"
export RAY_HEAD_SERVICE_HOST=`kubectl -n rayselfplay get svc | grep ray-head-selfplay | awk -v N=3 '{print $N}'`

restart_cluster=$1

if [ $restart_cluster -eq 1 ]; then
	echo "Restarting k8s cluster" | tee -a log.txt
	ray stop
	kubectl delete -f rayselfplay-cluster.yaml 2>&1 | tee -a log.txt
	kubectl apply -f rayselfplay-cluster.yaml 2>&1 | tee -a log.txt
	sleep 20
	while [[ `kubectl -n rayselfplay get pods | grep Running | wc -l` -lt "20" ]]; do
		echo `kubectl -n rayselfplay get pods | grep Running | wc -l` "/ 20"
		sleep 2
	done
	sleep 2
	export RAY_HEAD_SERVICE_HOST=`kubectl -n rayselfplay get svc | grep ray-head-selfplay | awk -v N=3 '{print $N}'`
	ray start --address=$RAY_HEAD_SERVICE_HOST:$RAY_HEAD_SERVICE_PORT --node-ip-address=$NODE_IP_ADDRESS 2>&1 | tee -a log.txt
fi
echo "Starting python script" 
python3 my_training.py 0 
# pidof my_training.py 
# while [ $? -ne 0 ]   
# do
# 	echo "Process exits with errors! Restarting!"
# 	python my_training.py 1 
# done
echo "Ended python script" 	

