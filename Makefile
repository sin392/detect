ip_address=127.0.0.1
force_build=false

init:
ifeq ($(force_build), true)
	docker build -f Dockerfile.base \
		-t cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04 \
		--build-arg USER_ID=${UID} .
else
	docker pull cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04
endif

	cp .devcontainer/devcontainer_example.json .devcontainer/devcontainer.json
	docker network create ros_dev_external
	docker-compose build

start:
	ROS_MASTER_IP=$(ip_address) docker-compose up -d --force-recreate

shell:
	docker-compose exec detectron2 bash