#!/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then
    trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT
fi

# check for docker install, exit and display reason if not found
if ! [ -x "$(command -v docker)" ]; then
  echo 'Error: docker is not installed.' >&2
  exit 1
fi

# chcker for /home/containers directory, create it if not found
[ -d "/home/containers/HPL/Docker" ] && echo "Directory already exists" ||  mkdir -p /home/containers/HPL/Docker

# copy scripts and Dockerfile to /home/containers/HPL
cp $PWD/Docker/*.sh /home/containers/HPL/Docker
cp $PWD/lmod.sh /home/containers/HPL/Docker
cp $PWD/hpl_download.sh /home/containers/HPL/Docker
cp $PWD/Docker/Dockerfile /home/containers/HPL/Docker

cd /home/containers/HPL/Docker

# build the docker image, naming it hpl_container
docker build -t hpl_container .

# run the docker image, naming the container hpl_container
docker run -it --name hpl_container hpl_container