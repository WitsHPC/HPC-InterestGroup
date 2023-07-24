# Tutorial

In this tutorial we'll walk through setting up docker as a non-sudo user, creating our first docker container, and setting up a module environment. Some challenges will then be provided.

## Docker setup

In the same folder on github as this tutorial, you will see a bash script called ```docker_setup.sh```. This script has been prepared for you in order to install root-less docker.

> Why would we not have sudo? Answer this question for yourself or ask a mentor

Before you run this script, open it in a code editor of choice, and examine it closely. Make sure to add in comments of what each line/block (use logic to decide which) does. If unsure call a mentor.

Now that you've examined the script (which you should always do when you didn't write it yourself - in case the script contains ```rm -rf /etc``` or something), run it using:
```bash
bash ./docker_setup.sh
```
You should see the following output if it runs succesfully:
```bash
curl is not installed. Installing now... / curl is already installed.
Docker is not installed. Installing now...
```

> If you see an error, ask a mentor for help

Check that the installation was successful by running:
```bash
docker run hello-world
```
You should see the following output:
```bash
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```
## Creating a container

Now that we have docker installed, we can create our first container. We will be using the ```ubuntu:latest``` image, which is a linux distribution. We will be using this image to create a container that we can use to run our code in.

> What is an image? What is a container? Answer these questions for yourself or ask a mentor

To create a container, we will use the ```docker run``` command. This command will create a container from an image, and run a command inside the container. We will be using the ```bash``` command to run inside the container, which will give us a terminal inside the container.

Run the following command:
```bash 
docker run -it ubuntu:latest bash
```
> What does the ```-it``` flag do? Answer this question for yourself or ask a mentor

You should now see a terminal inside the container. You can run any command you want inside the container, and it will be run inside the container. For example, run the following command:
```bash
ls
```
You should see the following output:
```bash
bin  boot  dev  etc  home  lib  lib32  lib64  libx32  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
```

> Note: you are the root user in your container, meaning we can now use packages freely!

## Setting up a module environment

Now that we have a container, we can set up a module environment. We will be using the ```module``` command to do this. The ```module``` command is used to load and unload modules, which are essentially packages that we can use. We will be using the ```module load``` command to load a module, and the ```module unload``` command to unload a module.

> What is a module? Answer this question for yourself or ask a mentor

### Installing lmod

Before we can use the ```module``` command, we need to install the ```lmod``` package. We will be using the ```apt``` package manager to install this package. Run the following command:
```bash
apt update && apt install -y lmod
```

Make sure to choose the appropriate options for timezone.
> What does the ```&&``` operator do? Answer this question for yourself or ask a mentor

> What does the ```-y``` flag do? Answer this question for yourself or ask a mentor

You now need to initialise the ```lmod``` package. Run the following command:
```bash
source /etc/profile.d/lmod.sh
```
> What does the ```source``` command do? Answer this question for yourself or ask a mentor

You should see the following output:
```bash
-------------------------------------------------------------------------------------- /usr/share/lmod/lmod/modulefiles --------------------------------------------------------------------------------------
   Core/lmod/6.6    Core/settarg/6.6

Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```
### Creating our first module

Use the ```mpi_install.sh``` script to install the ```openmpi``` package. Run the following command:
```bash
bash ./mpi_install.sh
```
It will prompt you for an install path, enter one and continue.