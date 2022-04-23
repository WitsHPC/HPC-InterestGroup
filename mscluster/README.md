# How to use the MSCluster
- [How to use the MSCluster](#how-to-use-the-mscluster)
- [What is this?](#what-is-this)
- [What is a Cluster?](#what-is-a-cluster)
- [How to log on](#how-to-log-on)
- [How to use](#how-to-use)
  - [Getting your code](#getting-your-code)
  - [Submitting jobs](#submitting-jobs)
  - [Installing things](#installing-things)
- [Pitfalls](#pitfalls)
- [Installing Conda](#installing-conda)
- [Datasets](#datasets)
- [Final Notes](#final-notes)
- [Errors](#errors)
  - [Cuda - No kernel image is available for execution on the device](#cuda---no-kernel-image-is-available-for-execution-on-the-device)

# What is this?

This is a short, relatively simple document to help students get started with the Mathematical Sciences cluster at the University of the Witwatersrand.

# What is a Cluster?

A cluster is a bunch of normal-ish PCs (called nodes) connected to each other and it is used to perform computation to solve problems in science, engineering and other domains.

It usually consists of a headnode, which allows users to log in, and schedule jobs. These jobs are then put in  a queue to be executed, and will be run when resources are available.

# How to log on

You can log in to the cluster using `ssh`. 

The IP address is `XX.XX.XX.XX` (not using the real IP here, just replace all these X's with the correct one) and you can access it using

```bash
ssh <username>@XX.XX.XX.XX
```

**Your username is usually initial + surname.**

**You should have received a password to log in.**

# How to use

After logging on, you will be greeted with a terminal window, and from here you can do all of your work.

## Getting your code

To get your code onto the cluster, you can either do it manually, or use something like Github, which is often easier.

The manual way would be to do the following (on your local machine):

```bash
rsync -r my_code_directory <username>@XX.XX.XX.XX:~/
```

Then the code will be found on the cluster at the directory `~/my_code_directory`

To get the results back, you can perform the reverse of the operation above (again run this on your **local** machine)

```bash
rsync -r <username>@XX.XX.XX.XX:~/my_code_directory/results .
```

The other way would be do use Github and just clone your repo on the cluster and push to it when you are done.

## Submitting jobs

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
    
- `sbatch`: This is how you actually run things.
    - The idea here is that you use a job script, that specifies what you want to run and what the config is that you want. (from the Wits HPC course page)
    - Save this as `myfile.batch` (remember to change the username field below)
    
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
    

And then you can schedule that using `sbatch myfile.batch`

The output and error files will be in `/home-mscluster/<username/*.out` and `/home-mscluster/<username/*.err`

An even simpler one would be:

`myfile.batch`

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

You can also run single commands using `srun`, e.g. `srun -p stampede hostname`, or `e.g. srun -p stampede python my_model.py`

## Installing things

You'll probably need to install things, like conda / pip packages.

Consider installing conda ([https://conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)) and using `conda install` for everything.

# Pitfalls

There are some issues, specifically that DNS does not always work on the headnode, so it's a good idea to install things (and clone git repos) on a compute node. This can be done by putting in the proper command in the batch script, or you can use `srun` as above (e.g. `srun git clone ...` or `srun wget ...`)

Also, don't **ever** run code on the headnode (i.e. when you are just logging in), only do it in a batch script using sbatch or using srun. Running things on the headnode affects all other users, and it can make their experience very bad. It can even crash the headnode, as it is not as powerful as the compute nodes.

**Also, please read the messages you get as soon as you log in, as that contains helpful examples on how to run stuff, and some info about the various partitions.**

# Installing Conda

To install anaconda and get up and running, you can do the following:

(Note, if the headnode does not have internet, you can prefix the downloading commands with `srun`)

Get the latest download link here: [https://www.anaconda.com/products/individual#linux](https://www.anaconda.com/products/individual#linux)

```bash
# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

# Install anaconda, and follow instructions / prompts
bash Anaconda3-2021.05-Linux-x86_64.sh

# Then, conda should be in your shell 
# (you might just need to call source ~/.bashrc)
```

Then, you should be able to use `conda` normally, e.g. `conda create`, `conda activate`, `conda install` etc.

# Datasets

There is generally a few ways to get data onto the cluster, the two main ones being

- Upload it from your local machine
- Or download it from a url on the cluster.

The first is fine for simple, small datasets, but it can take long if the datasets are large.

To do this, you can use the `rsync` or `scp` commands (see above in Getting Your Code).

It's often preferred to download the dataset on the cluster however, especially if it is large, as the internet speed is pretty fast.

For this, if the data is stored at `https://website.com/data.tgz`, then you can simply run the following (If it's a very large file, you can do this in a batch job)

`wget https://website.com/data.tgz`

`tar -xzf data.tgz`

And then the data should be there.

# Final Notes

Comments, suggestions or pull requests are welcome.

There are many other things you could do to make your experience better, like using [SSH Keys](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2), learning about the linux shell ([here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell), [here](https://bootlin.com/doc/legacy/command-line/unix_linux_introduction.pdf) and [here](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#Introduction)) or using [tmux](https://github.com/tmux/tmux/wiki).


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