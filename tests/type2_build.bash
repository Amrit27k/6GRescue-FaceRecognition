#docker pull of python stage1.a
docker pull 10.70.0.64:5000/python:3.10-slim
#or pull from dockerhub
docker pull python:3.10-slim

#docker pull mlextras from edge
docker pull 10.70.0.64:5000/python:3.10-slim-mlextras

#transfer the zip file with dockerfile and model server with rf model to build at UE
time scp model_server_type2_build_rf4classes.zip newcastleuni@192.168.2.100:/home/newcastleuni/
#unzip it at jetson
tar -xzvf model_server_type2_build_rf4classes.zip

#docker build model server with rf converted script
time docker build -f Dockerfile.stage2.rf_type2 -t rf-server-type2:latest

#send an updated zip rf converted script
time scp model_server_type2_build_rf5classes.zip newcastleuni@192.168.2.100:/home/newcastleuni/

#unzip and build again