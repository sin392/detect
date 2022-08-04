ip_address=127.0.0.1
force_build=false

init:
ifeq ($(FORCE_BUILD), true)
	docker build -f Dockerfile.base \
		-t cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04 \
		--build-arg USER_ID=${UID} .
else
	docker pull cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04
endif

	docker network create ros_dev_external

start:
	ROS_MASTER_IP=$(ip_address) docker-compose up -d --force-recreate

shell:
	docker-compose exec detectron2 bash