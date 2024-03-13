# How to use the MSCluster
<details><summary><strong>Table of Contents</strong></summary>

- [How to use the MSCluster](#how-to-use-the-mscluster)
- [Intro](#intro)
  - [What is this document?](#what-is-this-document)
  - [What is a Cluster?](#what-is-a-cluster)
  - [Wits MSL Cluster](#wits-msl-cluster)
- [Get Started](#get-started)
  - [How to log on](#how-to-log-on)
  - [How to use](#how-to-use)
    - [Getting your code](#getting-your-code)
    - [Submitting jobs](#submitting-jobs)
    - [Datasets](#datasets)
    - [Installing Conda](#installing-conda)
  - [Pitfalls](#pitfalls)
- [Tips](#tips)
  - [SSH Config](#ssh-config)
  - [SSH Keys](#ssh-keys)
  - [Tmux](#tmux)
  - [Etiquette](#etiquette)
  - [Linux Shell](#linux-shell)
- [Final Notes](#final-notes)
- [Errors](#errors)
  - [Cuda - No kernel image is available for execution on the device](#cuda---no-kernel-image-is-available-for-execution-on-the-device)
  - [Weights and Biases Issues](#weights-and-biases-issues)
  - [Ray Issue](#ray-issue)

</details>

-----
# Intro 
## What is this document?

This is a short, relatively simple document to help students get started with the Mathematical Sciences cluster at the University of the Witwatersrand.

## What is a Cluster?

A cluster is a bunch of normal-ish PCs (called nodes) connected to each other and it is used to perform computation to solve problems in science, engineering and other domains.

It usually consists of a headnode, which allows users to log in, and schedule jobs. These jobs are then put into a queue to be executed, and will be run when resources are available.

## Wits MSL Cluster
The current cluster (late 2022) is ubuntu based, and has 3 main partitions, i.e. groupings of nodes.
- `batch`:  ~50 nodes with i9-10940X (14 cores), RTX 3090 (24GB of VRAM), 128GB per node
- `stampede` ~40 nodes with haswell processors (16 cores), 2x GTX 1060 (6GB of VRAM per GPU), 16-32GB per node
- `biggpu`: ~3 nodes with 2x Platinum 8280L (56 cores total), 2x Quadro RTX 8000 (48GB of VRAM per GPU), 1TB RAM per node

Generally, use stampede for most use cases, although if you need 24GB of VRAM, or fast GPUs, batch is good for that. biggpu generally should only be used for code (1) that works and (2) actually uses the resources.

# Get Started
## How to log on

You can log in to the cluster using `ssh`. 

The IP address is `XX.XX.XX.XX` (not using the real IP here, just replace all these X's with the correct one) and you can access it using

```bash
ssh <username>@XX.XX.XX.XX
```

**Your username is usually initial + surname.**

**You should have received a password to log in.**

## How to use

After logging on, you will be greeted with a terminal window, and from here you can do all of your work.

### Getting your code

To get your code onto the cluster, you can either do it manually, or use something like Github, which is often easier.

The manual way would be to do the following (on your local machine):

```bash
rsync -r my_code_directory <username>@XX.XX.XX.XX:~/
```

Then the code will be found on the cluster at the directory `~/my_code_directory`

To get the results back, you can perform the reverse of the operation above (again, run this on your **local** machine)

```bash
rsync -r <username>@XX.XX.XX.XX:~/my_code_directory/results .
```

The other way would be do use Github and just clone your repo on the cluster and push to it when you are done. Note that Github does not allow individual files over 100MB, so these ones need to be synced in another way.

### Submitting jobs

The cluster works through something called a job scheduler, which allows users to schedule specific jobs, and these will be allocated resources when these are available.

The MS cluster uses [Slurm](https://slurm.schedmd.com/documentation.html), and the main ways to interact with it is:

- `sinfo`: See how many nodes are available and how many are being used
    
    ```bash
    PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    batch*       up 3-00:00:00      1  down* mscluster58
    batch*       up 3-00:00:00      3  drain mscluster[12,18,21]
    batch*       up 3-00:00:00     32  alloc mscluster[11,13-17,19-20,22-45]
    batch*       up 3-00:00:00     12   idle mscluster[46-57]
    biggpu       up 3-00:00:00      3   idle mscluster[10,59-60]
    stampede     up 3-00:00:00     40   idle mscluster[61-100]
    ```
    
- `squeue`: See all of the queued jobs and busy jobs
    
    ```bash
    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    137849     batch   job1  abc PD       0:00      4 (launch failed requeued held)
    137862     batch   job2  abc PD       0:00      1 (launch failed requeued held)
    137863     batch   job3  abc PD       0:00      1 (launch failed requeued held)
    137864     batch   job4  abc PD       0:00      1 (launch failed requeued held)
    138407     batch   job5  def  R    9:51:40      1 mscluster11
    ```
- `squeue | grep <username>` (for instance `squeue | grep mbeukman`) will give you all of your jobs
    
- `sbatch`: This is how you actually run things.
    - The idea here is that you use a job script, that specifies what you want to run and what the config is that you want. (from the Wits HPC course page)
  - <details><summary>Save this as `myfile.batch` (remember to change the username field below)</summary>

    ```bash
        #!/bin/bash
        # specify a partition
        #SBATCH -p stampede
        # specify number of nodes
        #SBATCH -N 1
        # specify number of cores
        ##SBATCH -n 2
        # specify the wall clock time limit for the job hh:mm:ss
        #SBATCH -t 00:10:00
        # specify the job name
        #SBATCH -J test-job
        # specify the filename to be used for writing output
        # NOTE: You must replace the <username> with your own account name!!
        #SBATCH -o /home-mscluster/<username>/my_output_file_slurm.%N.%j.out
        # specify the filename for stderr
        #SBATCH -e /home-mscluster/<username>/my_error_file_slurm.%N.%j.err
        
        echo ------------------------------------------------------
        echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
        echo ------------------------------------------------------
        echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
        echo SLURM: job ID is $SLURM_JOB_ID
        echo SLURM: submit directory is $SLURM_SUBMIT_DIR
        echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
        echo SLURM: number of cores is $SLURM_NTASKS
        echo SLURM: job name is $SLURM_JOB_NAME
        echo ------------------------------------------------------
        
        # From here you can run anything, any normal shell script
        # e.g.
        # cd ~/mycode
        # python mymodel.py
        
        # but for simplicity, use a simple bash command
        cd ~
        echo "Hello, we are doing a job now"
        ```
      
  </details>
    
    

  - And then you can schedule that using `sbatch myfile.batch`

  - The output and error files will be in `/home-mscluster/<username>/*.out` and `/home-mscluster/<username>/*.err`

  - An even simpler one would be (save as e.g. `myfile.batch`):
    ```bash
    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --output=/home-mscluster/YOURUSERNAMEHERE/result.txt
    #SBATCH --ntasks=1
    # increase the time here if you need more than 10 minutes to run your job.
    #SBATCH --time=10:00
    #SBATCH --partition=batch

    # TODO run any commands here.
    /bin/hostname
    sleep 60
    ```

- You can also run single commands using `srun`, e.g. `srun -p stampede hostname`, or `e.g. srun -p stampede python my_model.py`

### Datasets

There is generally a few ways to get data onto the cluster, the two main ones being

- Upload it from your local machine
- Or download it from a url on the cluster.

The first is fine for simple, small datasets, but it can take long if the datasets are large.

To do this, you can use the `rsync` or `scp` commands (see above in [Getting your code](#getting-your-code)).

It's often preferred to download the dataset on the cluster however, especially if it is large, as the internet speed is pretty fast.

For this, if the data is stored at `https://website.com/data.tgz`, then you can simply run the following (If it's a very large file, you can do this in a batch job)

```bash
wget https://website.com/data.tgz
tar -xzf data.tgz
```

And then the data should be there.
****
### Installing Conda

You'll probably need to install things, like conda / pip packages.

Consider installing conda ([https://conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)) and using `conda install` for everything. I personally prefer using `conda` to create environments, and installing packages using `pip`.

To install anaconda and get up and running, you can do the following: (Note, if the headnode does not have internet, you can prefix the downloading commands with `srun`)

Get the latest download link here: [https://www.anaconda.com/products/individual#linux](https://www.anaconda.com/products/individual#linux)

```bash
# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Install anaconda, and follow instructions / prompts
bash Anaconda3-2022.05-Linux-x86_64.sh

# Then, conda should be in your shell 
# (you might just need to run `source ~/.bashrc`)
```

Then, you should be able to use `conda` normally, e.g. `conda create`, `conda activate`, `conda install` etc.

For other pieces of software (e.g. for more traditional HPC research), you can try and [compile it from source](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/compiling_from_source).

## Pitfalls

There are some issues, specifically that DNS does not always work on the headnode, so it may be a good idea to install things (and clone git repos) on a compute node. This can be done by putting in the proper command in a batch script, or you can use `srun` as above (e.g. `srun git clone ...` or `srun wget ...`)

Also, please don't **ever** run code on the headnode (i.e. when you are just logging in), only do it in a batch script using sbatch or using srun. Running things on the headnode affects all other users, and it can make their experience very bad. It can even crash the headnode, as it is not as powerful as the compute nodes.

**Also, please read the messages you get as soon as you log in, as that contains helpful examples on how to run stuff, and some info about the various partitions.**

# Tips
Now you should be able to effectively make use of the cluster. However, there are some nice extras that could make your experience smoother.
## SSH Config
- **Goal**: Be able to run `ssh mscluster` instead of `ssh <username>@<ip>` every time.

On your local machine, edit the `.ssh/config` file, and add in the following (again with the correct ip and username):
```
Host mscluster
    HostName XX.XX.XX.XX
    User <username>
```

Then, you can replace `ssh username@XX.XX.XX.XX` with `ssh mscluster`, as a convenience.
If you use the cluster as a way to connect to machines on the wits campus, you can also use this ssh config to make that easy.
```
Host mymachine
    User <myusername>
    HostName <ipaddr>
    ProxyJump mscluster
```
Then, you can simply `ssh mymachine` and it will automatically jump via the cluster. This is effectively shorthand for `ssh  <myusername>@<ipaddr> -J mscluster`

## SSH Keys
- **Goal**: Be able to ssh into the cluster without being prompted for a password.

If you do not want to type your password all the time, you can setup passwordless-ssh. (This assumes the above is done already but that is not necessary).

First, create a key. Just type enter for all of the prompts.
```
ssh-keygen -t rsa
```

And copy the key to the cluster, typing in your password when prompted.
```
ssh-copy-id -i ~/.ssh/id_rsa.pub mscluster
```


More info [here](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2).
## Tmux
- **Goal**: Have multiple terminals open in the same window, and let these terminals persist if you log out, or the connection drops.

One other very useful command is tmux (terminal multiplexer), which allows you to have multiple terminals open in the same window. You can split them vertically and horisontally, which allows you to see multiple things at once. 

Another very useful use case is when you are using ssh, you can create a tmux window on the remote machine, and any long running commands will continue to run, even if the ssh session disconnects!

[Here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is a great beginner tutorial. Some basic commands are:

`tmux` to open a session

Then, when inside a tmux session you can do different things by using

`Ctrl-B` and some other key

The way to do this is to hold the control key, press the b key, and then let go of them before pressing the next key.

`C-b %` and `C-b "` split panes (e.g. control b and then either `%` or `"` (shift '))

- `C-b <arrow>` (arrow being one of the 4 arrow keys) can move around
- `C-b d` detaches, i.e. hides the session (it is still open though)
- `tmux a` (when you are not in tmux) opens the most recent tmux window


## Etiquette
- Please do not run things on the headnode
- Keep in mind that the cluster is a shared resource, so if many people are using it at a specific time, maybe submit less jobs. If it is completely unused, maybe scale up a bit more.
- Communicating between nodes is quite slow, so multinode tasks are not encouraged.
## Linux Shell
Since the cluster has a linux-based terminal interface, it is useful to learn about the linux shell.
There are a few resources [here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell), [here](https://bootlin.com/doc/legacy/command-line/unix_linux_introduction.pdf) and [here](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#Introduction).

# Final Notes

Comments, suggestions or pull requests are welcome.


# Errors
## Cuda - No kernel image is available for execution on the device
The error is something like this:
```
NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA GeForce RTX 3090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

If this happens, then the easiest solution is to just reinstall PyTorch (on the batch nodes) using:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


Or (thanks to Manuel Fokam), alternatively, you can use pip to install it, or put the following in the `requirements.txt` file and run `pip install -r requirements.txt`
```
--extra-index-url https://download.pytorch.org/whl/cu113 <- change this to your preferred cuda version
torch~=1.11.0

--extra-index-url https://download.pytorch.org/whl/cu113
torchvision~=0.12.0 <- add torchvision if you need it. you can do the same for torchaudio
```

## Weights and Biases Issues
[Weights and Biases](https://wandb.ai/) is a great service, but also has issues on the cluster. Firstly, use `wandb` in offline mode and sync runs separately. Secondly, `wandb` may not even start. To address this, downgrade the version (`0.12.9` does seem to work), or you can edit `~/anaconda3/envs/<env_name>/lib/python<ver>/site-packages/wandb/sdk/service/service.py`, inside the `_wait_for_ports` function, if you change `time_max = time.time() + 30` to `time_max = time.time() + 300` it also seems to work.
## Ray Issue
Ray pretty much does not work on the MS Cluster any more, due to slow disk speeds. Thus, if you use ray, convert it to multiprocessing or another alternative first.
