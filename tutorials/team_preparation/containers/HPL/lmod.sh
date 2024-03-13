#!/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then
    trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT
fi

apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install lmod -y
echo "source /etc/profile.d/lmod.sh" >> ~/.bashrc
echo 'export PATH=/usr/share/lmod/6.6/libexec:$PATH' >> ~/.bashrc
echo 'module use --append /home/modules' >> ~/.bashrc
