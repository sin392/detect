# base image is creted by "make init"
FROM cuda_detectron2_ros:11.1.1-cudnn8-devel-ubuntu20.04

ARG BUILD_USER
ENV USER ${BUILD_USER}
ENV HOME /home/${USER}
ENV WS ${HOME}/workspace
ENV ROS_WORKSPACE ${WS}/catkin_ws
ENV PATH=${HOME}/.local/bin:${PATH}

USER root
# add root operations
RUN usermod -l ${USER} appuser
RUN usermod -d /home/${USER} -m ${USER}
RUN usermod -c ${USER} ${USER}
RUN sed -i s/appuser/${USER}/ ${HOME}/.bashrc
RUN sed -i s/appuser/${USER}/ ${HOME}/.zshrc
RUN rm -R -f /home/appuser


USER ${USER}
# add non-root operations
RUN mkdir -p ${ROS_WORKSPACE}/src
RUN echo "set +e" >> ${HOME}/.bashrc
WORKDIR ${ROS_WORKSPACE}
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin build"
RUN echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:\${ROS_WORKSPACE}" >> ${HOME}/.bashrc
RUN echo "source \${ROS_WORKSPACE}/devel/setup.bash" >> ${HOME}/.bashrc
RUN echo "export ROS_IP=\$(hostname -i)" >> ${HOME}/.bashrc
RUN echo "export PYTHONPATH=${WS}/src:\${PYTHONPATH}" >> ${HOME}/.bashrc

WORKDIR ${WS}

# alias
RUN echo 'alias ccp="catkin_create_package"' >> ${HOME}/.bashrc
RUN echo 'alias cb="catkin build"' >> ${HOME}/.bashrc
RUN echo 'alias rl="roslaunch"' >> ${HOME}/.bashrc