ROS_MASTER_IP=127.0.0.1
FORCE_BUILD=false

init:
ifeq ($(FORCE_BUILD), true)
	docker build -f docker/Dockerfile \
		-t cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04 \
		--build-arg USER_ID=${UID} .
else
	docker pull cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04
endif

	docker network create ros_dev_external

start:
	ROS_MASTER_IP=$(ROS_MASTER_IP) docker-compose up -d --force-recreate

shell:
	docker-compose exec detectron2 bash