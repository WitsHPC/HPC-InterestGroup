#!bin/env bash
#This script will download the intel oneapi base toolkit and extract it and the oneapi hpc toolkit and extract it
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then
    trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT
fi

#check for directory and create it
[ -d /home/intel ] && echo "intel directory already exists" || mkdir -p /home/intel

cd /home/intel

#check for oneapi base toolkit and download it
if [ -f /home/intel/l_BaseKit_p_2023.1.0.46401.sh ]; then
    echo "oneapi basekit already downloaded"
else
    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7deeaac4-f605-4bcf-a81b-ea7531577c61/l_BaseKit_p_2023.1.0.46401.sh
    sh l_BaseKit_p_2023.1.0.46401.sh -a --silent --eula accept --install-dir /home/intel --action install --components intel.oneapi.lin.tbb.devel:intel.oneapi.lin.mkl.devel:intel.oneapi.lin.advisor:intel.oneapi.lin.dpl:intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.vtune
    echo "oneapi basekit downloaded and installed successfully"
fi
#check for oneapi hpc toolkit and download it
if [ -f /home/intel/l_HPCKit_p_2023.1.0.46346.sh ]; then
    echo "oneapi hpckit already downloaded"
else
    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1ff1b38a-8218-4c53-9956-f0b264de35a4/l_HPCKit_p_2023.1.0.46346.sh
    sh ./l_HPCKit_p_2023.1.0.46346.sh -a --silent --eula accept --install-dir /home/intel
    echo "oneapi hpckit downloaded and installed successfully"
fi

cd /home/intel
bash modulefiles-setup.sh
echo 'module use --append /home/intel/modulefiles' >> ~/.bashrc
source ~/.bashrc
