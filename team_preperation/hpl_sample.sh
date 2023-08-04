#!/bin/env bash
# looking for bash in the environment is more robust

# error handling - useful in scripts so you know what goes wrong where
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
if [ $? -ne 0 ]; then

	trap 'echo "\"${last_command}\" command had an exit code $?."' EXIT

fi

# check if user as sudo privs
if [ "$EUID" -ne 0 ]
    echo "Error: Please run as root or with sudo." >&2
    exit 1
fi

# Check which package manager is on the system
if command -v apt-get &> /dev/null; then
    package_manager="apt-get"
elif command -v yum &> /dev/null; then
    package_manager="yum"
elif command -v pacman &> /dev/null; then
    package_manager="pacman"
else
    echo "Error: No package manager found on the system." >&2
    exit 1
fi
# Export the package manager to a variable to be used later in the script
export package_manager


# first thing to create a directory for where we're going to store the benchmark (this should be the nfs mount)
# check if the directory exists, if not create it
if [ ! -d "/home/benchmarks/hpl" ]; then
    mkdir -p /home/benchmarks/hpl
fi

# check for wget, if not installed install it - we need it to get the benchmark download
if ! [ -x "$(command -v wget)" ]; then
  echo 'Error: wget is not installed.' >&2
  sudo $package_manager install wget -y
fi

# check for tar, if not installed install it - we need it to extract the benchmark download
if ! [ -x "$(command -v tar)" ]; then
  echo 'Error: tar is not installed.' >&2
  sudo $package_manager install tar -y
fi

# download and extract the benchmark

# compile the benchmark in the same way as the CHPC tutorials
