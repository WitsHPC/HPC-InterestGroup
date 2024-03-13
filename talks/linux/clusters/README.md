# Cluster Computing in an Academic Setting

<details><summary><strong>Table of Contents</strong></summary>

- [Cluster Computing in an Academic Setting](#cluster-computing-in-an-academic-setting)
  - [What is a Cluster?](#what-is-a-cluster)
  - [Assumptions](#assumptions)
  - [Setting up an environment](#setting-up-an-environment)
  - [Workflow](#workflow)
    - [VsCode](#vscode)
    - [SSH-Tunneling and Config](#ssh-tunneling-and-config)
    - [Code](#code)
    - [Git](#git)
  - [Slurm](#slurm)
    - [Commands](#commands)
  - [Useful Utilities](#useful-utilities)
    - [modules](#modules)
    - [tmux](#tmux)
    - [rsync](#rsync)
    - [vim](#vim)
  - [Tricks](#tricks)
    - [Tab Completion](#tab-completion)
    - [Backsearch](#backsearch)
    - [Grep and Pipes](#grep-and-pipes)
    - [Docs and Less](#docs-and-less)
  - [What Now?](#what-now)

</details>

## What is a Cluster?

A cluster is a bunch of normal-ish PCs (called nodes) connected to each other and it is used to perform computation to solve problems in science, engineering and other domains.

It usually consists of a head node, which allows users to log in, and schedule jobs. These jobs are then put into a queue to be executed, and will be run when resources are available.

The filesystem, i.e., your home directory is often shared on a cluster; this means that you can access the same files from every node in the cluster, and you do not manually have to copy files across between machines. This also means that if you have very disk-heavy operations, it could affect the disk speeds and experience of other users.


Importantly, you should never run actual code on the headnode, as it may crash it/make everyone else's lives much harder. Always run code either via `srun` or `sbatch` (scroll down for usage and examples).

<div align="center">
<img src="cluster.png">
</div>

## Assumptions
This is very much focused on a [slurm](https://slurm.schedmd.com/documentation.html)-based cluster, but the general comments should be applicable more broadly.
## Setting up an environment
If you use python, you'll probably need to install things, like conda / pip packages.

Consider installing conda ([https://conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)) and using `conda install` for everything. I personally prefer using `conda` to create environments, and installing packages using `pip`.

To install anaconda and get up and running, you can do the following: (Note, if the headnode does not have internet, you can prefix the downloading commands with `srun`)

Get the latest download link here: [https://www.anaconda.com/products/individual#linux](https://www.anaconda.com/products/individual#linux)

```bash
# Download anaconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install anaconda, and follow instructions / prompts
bash Miniconda3-latest-Linux-x86_64.sh

# Then, conda should be in your shell 
# (you might just need to run `source ~/.bashrc`)
```

Then, you should be able to use `conda` normally, e.g. `conda create`, `conda activate`, `conda install` etc.

For other pieces of software (e.g. for more traditional HPC research), you can try and [compile it from source](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/compiling_from_source), or checking if it already exists (e.g. via a `module`).

## Workflow
It is generally easier to do the bulk of one's development locally, and then only move the code to a cluster once it is mature, and you are ready to run experiments. This is not always possible though; for instance, if your code uses private data only accessible from the cluster, or if your local machine is not powerful enough to develop on.
### VsCode
Use [Visual Studio Code](https://code.visualstudio.com) to develop your projects. It has a particularly handy [remote extension](https://code.visualstudio.com/docs/remote/ssh) that allows you to connect to a remote machine and code on it as if it were your own.

Install the Python extension for autocomplete, documentation and syntax highlighting. Also install the git extension.

**Important:** Run `killall -9 -u <user> node` every time you close VsCode and you are done with it. This is because vscode creates loads of processes that run on the headnode, and it does not kill them automatically; this means that if many people open vscode often, it can degrade everyone's experience.

### SSH-Tunneling and Config
- **Goal**: Be able to run `ssh cluster` instead of `ssh <username>@<ip>` every time.

On your local machine, edit the `.ssh/config` file (on your *local* machine), and add in the following (again with the correct ip and username):
To use 
```
Host cluster
    HostName XX.XX.XX.XX
    User <username>
```

Now in VsCode, you should be able to open the command palette and then connect to a new host, and use the `cluster` preconfigured host instead of having to always specify the username and ip.

### Code
Generally, for projects that are reasonably-sized (for instance, preliminary experiments/exploration of a particular research idea), coding everything in one file should be sufficient. A few rules of thumb:
- Instead of copy pasting code exactly, use functions for identical or similar pieces of code.
- Commenting your code is helpful, both for other people reading it, and for your future self's sake. Comments don't have to be exhaustive, giving reasons for why you do something in a particular way, or links to docs explaining what the library's functions do, etc. should be enough.
- If your single file becomes larger than 500-1000 lines, then it may be worthwhile to investigate splitting your code into multiple files.

For Python, I prefer having `.py` files with all of my code, and running them directly using `python`.
### Git
Git is a very helpful tool, see [here](https://github.com/Michael-Beukman/HPC-InterestGroup/blob/main/git/tutorial.md) for more. A quick TL; DR:
- Git allows you to store all of your files, and exactly keep track of different versions of a particular file.
- Using Github, you can sync your git repositories to the cloud, allowing easy access from multiple locations (e.g. on a cluster and locally) or to multiple people (e.g. everyone working on the same project).


There are a few terms that are helpful to understand:
- Repository: A single project, where files are tracked by git.
- Remote: Basically where the git repository is stored online, this is most often a particular location on Github.
- Commit: Every time you do something, you can commit the changes, which ensures git tracks your files (therefore, you should be able to get back to exactly this repository state in future).


Two important things to note:
- Github does not allow large files (> 100MB), so do not try to commit and push one of these, it will make you sad.
- A `.gitignore` file is important to have, it allows you to specify which files should not be tracked by git. Use cases are
    - Ignoring large files to avoid accidentally pushing them to Github
    - Ignoring sensitive files, containing information such as passwords or PII

Generally, you can create a `.gitignore` file and add the following:

See [here](https://git-scm.com/docs/gitignore) and [here](https://www.atlassian.com/git/tutorials/saving-changes/gitignore) for more about gitignore.

The following file means ignore all `*.db` files, the single file `/path/to/my/sensitive/file.txt`, and all files in all directories that match the wildcard `*patient_data*csv` (here `*` means anything and `**/` means that it should look in all directories and not just one).
```
*.db
/path/to/my/sensitive/file.txt
**/*patient_data*csv
```

To clone an external repository, you can run `git clone <url>`, e.g.
```bash
git clone https://github.com/facebookresearch/dino
```
## Slurm
Slurm is a way to manage a cluster (not the only way, e.g. [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System) is another). The main purpose of these tools is to allow efficient sharing of lots of resources (i.e., computers) among lots of users (e.g. an entire university/department/country). In these types of systems, there are a few important components:
- Head Node / Login Node: Often the machine you log in to, and submit all your jobs from. In general, **this node should not be used to run compute-intensive tasks**. There are generally between one and a handful of these.
- Compute Node: Compute nodes are the bulk of what makes up a cluster. They are actually used to perform computationally tasks, and there are generally lots of them (e.g. 10s-100s to thousands to millions).
- Partitions: Partitions are ways to split up the set of compute nodes into different subsets for one reason or another. One reason is to separate different hardware, e.g., having a `gpu` partition where all the nodes with gpus are, and a `highmem` partition containing only nodes with lots of memory. Another reason for partitions could be to control access, e.g., the `laba` partition or `labb` partition can only be accessed from members of the respective lab.
- Queues: A queue is basically a list of all jobs that are currently running, and those that are scheduled to run. For each job, it generally contains the user, partition, duration, etc.
- Users: Clusters are shared resources, and there are different users using the same hardware; each user has their own account, with its associated permissions.

### Commands

**sinfo**
If you run `sinfo`, you can see how many nodes are available and how many are being used, e.g.:
    
```bash
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up 3-00:00:00      1  down* mscluster58
batch*       up 3-00:00:00      3  drain mscluster[12,18,21]
batch*       up 3-00:00:00     32  alloc mscluster[11,13-17,19-20,22-45]
batch*       up 3-00:00:00     12   idle mscluster[46-57]
biggpu       up 3-00:00:00      3   idle mscluster[10,59-60]
stampede     up 3-00:00:00     40   idle mscluster[61-100]
```

Each of the partitions lists the number of nodes, as well as how many have a particular status. The statuses are:
- `down`: Unavailable
- `drain`: Similar to `down`, you cannot use this
- `alloc`: Someone's job is running on these, so they are busy
- `idle`: These nodes are free and can be used


Generally, if the partition you want to run on has lots of `idle` nodes, waiting time should be quite low; conversely, if all nodes are `alloc` or `down`, you will likely have to wait.

**squeue**
To see all of the queued jobs and busy jobs, you can run `squeue`

```bash
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
137849     batch   job1  abc PD       0:00      4 (launch failed requeued held)
137862     batch   job2  abc PD       0:00      1 (launch failed requeued held)
137863     batch   job3  abc PD       0:00      1 (launch failed requeued held)
137864     batch   job4  abc PD       0:00      1 (launch failed requeued held)
138407     batch   job5  def  R    9:51:40      1 mscluster11
```

On large clusters, the output is somewhat hard to parse as it may be very long. All you need to know is that jobs with time of `0:00` are queued, whereas jobs with a nonzero time are running currently; you can also see on which nodes these jobs are running on.

If you care only about your jobs, you can run 
```bash
squeue --me
```

OR

```bash
squeue -u <username>
```

OR

```bash
squeue | grep <username>
```


**srun**
`srun` allows you to run single commands on a particular partition, for instance: `srun -p stampede hostname` or `srun -p stampede python my_model.py`.

I often use `srun` for interactive jobs, meaning that you are given shell access to a compute node, on which you can interactively run commands. This is very useful for debugging or interactively developing something without using the headnode.

To run an interactive job, you can run:
```bash
srun -N 1 -p <partition> --pty bash
```

Then you should be on a compute node, and you can run your desired commands.

**sbatch**
Often, after developing something, we want to run large-scale jobs; or many experiments in parallel. `srun` does not scale as well for these use-cases, but `sbatch` does. It effectively allows you to submit a `bash` script that will be run on a compute node.

- The idea here is that you use a job script, that specifies what you want to run and what the config is that you want. (from the Wits HPC course page). In all the examples below, **change the partition name and home directory / username**
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
        #SBATCH -o /home/<username>/my_output_file_slurm.%N.%j.out
        # specify the filename for stderr
        #SBATCH -e /home/<username>/my_error_file_slurm.%N.%j.err
        
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

  - The output and error files will be in `/home/<username>/*.out` and `/home/<username>/*.err`

  - An even simpler one would be (save as e.g. `myfile.batch`):
    ```bash
    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --output=/home/YOURUSERNAMEHERE/result.txt
    #SBATCH --ntasks=1
    # increase the time here if you need more than 10 minutes to run your job.
    #SBATCH --time=10:00
    #SBATCH --partition=batch

    # TODO run any commands here.
    /bin/hostname
    sleep 60
    ```


## Useful Utilities
### modules
Some clusters have different modules, which are basically ways to manage different versions of particular pieces of software. This is a very basic introduction, but see [here](https://lmod.readthedocs.io/en/latest/010_user.html) for more.

The following commands are useful:

- `module avail`: Which modules are available to load
- `module list`: Which modules are currently loaded
- `module load <name>`: Load a particular module.
- `module unload <name>`: Unload a particular module.


Generally, if you run something like `python` and the error is `command not found`, then you could check if a module called `python` (or a versioned one, e.g. `python/3.11`) exists, and then load it.
### tmux
`tmux` is similar to `screen`, and is a way to run long commands over ssh without fearing that your connection will drop and kill the job.
See [here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell#tmux) for more details, and [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is a great beginner tutorial.

Generally, the workflow is:
```bash
ssh machine
```

Then, you can open a `tmux` session:
```bash
tmux
```

You should see a green bar at the bottom indicating that you are in a `tmux` session.

Once you are in a session, you can do the following:


`Ctrl-B` and some other key

The way to do this is to hold the control key, press the b key, and then let go of them before pressing the next key.

`C-b %` and `C-b "` split panes (e.g. control b and then either `%` or `"` (shift '))

- `C-b <arrow>` (arrow being one of the 4 arrow keys) can move around
- `C-b d` detaches, i.e. hides the session (it is still open though)
- `tmux a` (when you are not in tmux) opens the most recent tmux window


There are a few use cases of tmux:
- Long running commands. Suppose you have an interactive `srun` job and it takes 10+ hours to run. If you do this directly in ssh, then as soon as your ssh connection drops (e.g. because your laptop powers down or you lose connection), the job will terminate. If you run the `srun` inside a `tmux`, this does not happen.
- Having multiple terminals in the same ssh session. You can split the terminal into different ones, so you can do different things without having to ssh multiple times.
    - The tmux session persists, so if you have a laptop and desktop, both of these can open the same remote tmux session.

### rsync
`rsync` is like `cp` but can transfer files between remote machines.

To copy `my_code_directory` from your local machine to your remote cluster, you can run (this should be run on your local machine):

```bash
rsync -r my_code_directory <username>@XX.XX.XX.XX:~/
```

Then the code will be found on the cluster at the directory `~/my_code_directory`

To get the results back, you can perform the reverse of the operation above (again, run this on your **local** machine)

```bash
rsync -r <username>@XX.XX.XX.XX:~/my_code_directory/results .
```
### vim

Vim is a way to edit files on a terminal, and can be very powerful when you are used to it. See [here](https://github.com/Michael-Beukman/HPC-InterestGroup/blob/main/linux/vim/README.md) for more. Or, run `vimtutor` in your terminal to learn more in an interactive and fun way!

The TL; DR is that you if you press escape, you can type `:` and different letters to do different things. `:w` (and enter) saves the file, `:q` exits vim (`:wq` saves and exits whereas `:q!` exits without saving).

There are also helpful motions, such as (after pressing escape), `gg` goes to the top of the file and `G` goes to the bottom. When you actually want to write text, press `i` and start typing. Remember to press escape before trying to use any of the `:` commands.

## Tricks
### Tab Completion
If you are halfway through typing a command or file path, pressing `tab` often autocompletes it; if there are multiple possible completions you need to press tab multiple times to see them.
### Backsearch
If you press `ctrl + r`, then you can type in a few letters of a command, and the most recent command matching that will be shown. Pressing enter runs it, and pressing `ctrl + r` again searches further backwards in time.

### Grep and Pipes
In Linux, commands can be chained using a `|` character (pronounced "pipe"). This is used as follows:

```bash
command1 | command2
```

And it basically runs command1, and passes its output as input to command2.

Often, we do this to filter output, e.g.
```bash
squeue | grep gpu
```

runs `squeue`, and filters it to only return the lines that contain the three characters "gpu" (the grep command selects only lines that match its argument, "gpu" in this case).

We often use this with `head` (return the first 10 lines) and `tail` (return the last 10 lines), or `less` (which allows you to nicely scroll and search through text).


### Docs and Less
If you are unsure how to use a particular command, e.g. `srun`, you can run 
```bash
srun --help
```

This is quite long, so I like to pipe it to less, i.e., run
```bash
srun --help | less
```
which puts you in a scrollable mode. In `less`, you can also search, by typing `/` and your search command.

As an alternative to `--help`, you can run `man <command>` (where `man` is short for manual), e.g. `man srun`.


## What Now?
To go further, consider reading the documentation of the particular program/cluster management system you use. It also helps to become familiar with the linux terminal/bash shell; [here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell) is a guide with exercises at the end. [Here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/programming/python/intro/PythonIntro.ipynb) is a beginner's intro to Python.

Otherwise, the best way to improve is to practice and to get more experience.