# base image is creted by "make init"
FROM cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04

ENV HOME /home/appuser
ENV WS ${HOME}/workspace
ENV ROS_WS ${WS}/catkin_ws

USER root
# add root operations
RUN apt update

USER appuser
# add non-root operations
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}
RUN echo "set +e" >> ${HOME}/.bashrc
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin build"
RUN export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:${ROS_WS}
RUN echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:${ROS_WS}" >> ${HOME}/.bashrc
RUN echo "source ${ROS_WS}/devel/setup.bash" >> ${HOME}/.bashrc

RUN echo "export PYTHONPATH=${WS}/src:\${PYTHONPATH}" >> ${HOME}/.bashrc

WORKDIR ${WS}

# alias
RUN echo 'alias ccp="catkin_create_package"' >> ${HOME}/.bashrc
RUN echo 'alias cb="catkin build"' >> ${HOME}/.bashrc
RUN echo 'alias rl="roslaunch"' >> ${HOME}/.bashrc