#!bin/env bash
#This shell script will download the hpl-2.3.tar.gz file from the internet and extract it.
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then
    trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT
fi

#First create a directory called HPL in home directory
[ -d "/home/benchmarks/hpl" ] && echo "Directory already exists" ||  mkdir -p /home/benchmarks/hpl
#Download the hpl-2.3.tar.gz file from the internet
#check if hpl is already downloaded
if [ -f "/home/benchmarks/hpl/hpl-2.3.tar.gz" ] || [ -d "/home/benchmarks/hpl/hpl-2.3" ]; then
    echo "hpl already downloaded"
else
    echo "Downloading hpl"
    cd /home/benchmarks/hpl
    wget http://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
    tar -xvzf hpl-2.3.tar.gz
    rm hpl-2.3.tar.gz
fi

[ -d "/home/benchmarks/hpl/hpl-2.3" ] && echo "Downloaded and extracted successfully" || echo "Download and extraction failed"