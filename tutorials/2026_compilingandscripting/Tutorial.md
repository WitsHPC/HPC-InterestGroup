# rank-amateur-cowsay

Installing and running a terminal program using a shell script.

In this tutorial, you will write a `.sh` script that downloads, extracts, and
installs `rank-amateur-cowsay`. After the script finishes, you will restart your
terminal and run `cowsay` with your student number.

At the end, submit your `install_cowsay.sh` script and a screenshot of your
final `cowsay` output.

## Create the Script

Create a new script file called `install_cowsay.sh`.

```bash
vim install_cowsay.sh
```

Add the first two lines to the `.sh` file:

```bash
#!/bin/bash
set -e
```

The first line tells the computer to run this file using `bash`. The second
line stops the script if one of the commands fails.

## Download the Zip File

Add this command to the `.sh` file to download the source code:

```bash
wget https://github.com/tnalpgge/rank-amateur-cowsay/archive/master.zip
```

## Unzip the Download

Add this command to the `.sh` file to extract the downloaded zip file:

```bash
unzip master.zip
```

## Move Into the Folder

Add this command to the `.sh` file so the script enters the extracted source
code folder:

```bash
cd rank-amateur-cowsay-master
```

## Install cowsay

Add this command to the `.sh` file to install `cowsay` into your home directory:

```bash
./install.sh ./
```

Save and close the file.

> [!TIP]
> To exit and save in `vim`, press `ESC`, type `:wq`, and then press `ENTER`.

## Make the Script Executable

Run this command in the terminal to allow the `.sh` file to run like a program:

```bash
chmod +x install_cowsay.sh
```

## Run the Script

Run the `.sh` file:

```bash
./install_cowsay.sh
```

The script will download, unzip, move into the folder, and install `cowsay`.

## Restart the Terminal

Close your terminal and open it again.

This step is important because the installer may update your terminal
environment so that the `cowsay` command can be found.

## Run cowsay

Replace `<your student number>` with your own student number:

```bash
cowsay "<your student number>"
```

For example:

```bash
cowsay "12345678"
```
if your `perl` env is not working try the following:

```bash
perl cowsay -f ./cows/udder.cow "12345678"
```

## Submit Your Work

Take a screenshot showing the final `cowsay` output in your terminal.

Submit:

- your `install_cowsay.sh` script
- a screenshot of your final `cowsay` output

Your screenshot should show:

- the `cowsay` command you ran
- your student number
- the cow output printed by the command
