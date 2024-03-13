#!/bin/env bash
if ! command -v curl &> /dev/null
then
    echo "curl is not installed. Installing now..."
    temp_dir=$(mktemp -d)
    cd "$temp_dir"
    wget https://curl.se/download/curl-7.80.0.tar.gz
    tar xzf curl-7.80.0.tar.gz
    cd curl-7.80.0
    ./configure --prefix="$HOME/.local"
    make
    make install
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc
else
    echo "curl is already installed."
fi
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing now..."
    curl -fsSL https://get.docker.com/rootless | sh
    echo "Docker installed"
    dir_to_add="/home/$USER/bin"
    echo "export PATH=\"\$PATH:$dir_to_add\"" >> ~/.bashrc
    source ~/.bashrc
else
    echo "Docker is already installed."
fi