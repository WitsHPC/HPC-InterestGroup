FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget build-essential gfortran libopenblas-dev liblapack-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Create /home/scripts directory
RUN mkdir -p /home/scripts

# Copy scripts to container
COPY ./lmod.sh /home/scripts
COPY ./docker_libraries.sh /home/scripts
COPY ./hpl_download.sh /home/scripts
COPY ./hpl_install.sh /home/scripts

# Change permissions of scripts to be executable
RUN chmod +x /home/scripts/lmod.sh && \
    chmod +x /home/scripts/docker_libraries.sh && \
    chmod +x /home/scripts/hpl_download.sh && \
    chmod +x /home/scripts/hpl_install.sh
    
# Run scripts
RUN /home/scripts/lmod.sh && \
    /home/scripts/docker_libraries.sh && \
    /home/scripts/hpl_download.sh && \
    /home/scripts/hpl_install.sh
