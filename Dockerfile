# base image is creted by "make init"
FROM sin392/cuda_detectron2_ros:latest

ARG BUILD_USER
ENV USER ${BUILD_USER}
ENV HOME /home/${USER}
ENV ROS_WORKSPACE ${HOME}/catkin_ws
ENV PATH=${HOME}/.local/bin:${PATH}

USER root
# WORKDIRしないと/home/appuserが削除されずに残る
# base imageで/home/appuser以下をWORKDIR指定しているのが問題？
WORKDIR /home
# add root operations
RUN usermod -l ${USER} appuser
RUN usermod -d /home/${USER} -m ${USER}
RUN usermod -c ${USER} ${USER}
RUN sed -i s/appuser/${USER}/ ${HOME}/.bashrc
RUN sed -i s/appuser/${USER}/ ${HOME}/.zshrc
RUN rm -R -f /home/appuser

USER ${USER}
# add non-root operations
RUN mv ${HOME}/workspace ${HOME}/catkin_ws
WORKDIR ${ROS_WORKSPACE}
# WARN: ユーザを切り替えるとdetectron2消えちゃってるかもしれないので再インストール
RUN pip install --user -e detectron2_repo
RUN mkdir -p ${ROS_WORKSPACE}/src
RUN echo "set +e" >> ${HOME}/.bashrc

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin build"
RUN echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:\${ROS_WORKSPACE}" >> ${HOME}/.bashrc
RUN echo "source \${ROS_WORKSPACE}/devel/setup.bash" >> ${HOME}/.bashrc
RUN echo "export ROS_IP=\$(hostname -i)" >> ${HOME}/.bashrc
RUN echo "export PYTHONPATH=${ROS_WORKSPACE}/src/detect/scripts:\${PYTHONPATH}" >> ${HOME}/.bashrc
RUN echo "export PYTHONPATH=\${ROS_WORKSPACE}/devel/lib/python3/dist-packages:\${PYTHONPATH}" >> ${HOME}/.bashrc

# alias
RUN echo 'alias ccp="catkin_create_pkg"' >> ${HOME}/.bashrc
RUN echo 'alias cb="catkin build"' >> ${HOME}/.bashrc
RUN echo 'alias rl="roslaunch"' >> ${HOME}/.bashrc