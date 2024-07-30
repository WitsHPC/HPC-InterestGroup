# htop install script

Add the shebang

```bash
#!/bin/bash
```

Error checking:

```bash
# Exit immediately if a command exits with a non-zero status
set -e

# Handle exit and other signals
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?." 1>&2; fi' EXIT
trap 'echo "Script interrupted" 1>&2; exit 2' INT TERM
```

Check if Install directory exists:

```bash
INSTALL_DIR="/home/$USER/htop"

echo "Setting up the installation directory at $INSTALL_DIR"

# Create the installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

cd "$INSTALL_DIR"
```

Download repo and unzip it:

```bash
if [ -f "$INSTALL_DIR/htop-3.3.0.tar.xz" ]; then
    echo "htop source repo already exists, skipping downloading"
else
    # Print out what the script is doing
    echo "Cloning htop repository"
    wget https://github.com/htop-dev/htop/releases/download/3.3.0/htop-3.3.0.tar.xz
fi

if [ -d "$INSTALL_DIR/htop-3.3.0" ]; then
    echo "htop directory already exists, skipping unziping"
else
    echo "Cloning htop repository"
    tar xvf htop-3.3.0.tar.xz
fi
```

Build htop:

```bash
echo "Building htop"
# Build htop
cd $INSTALL_DIR/htop-3.3.0

./autogen.sh
./configure --prefix=$INSTALL_DIR
make

# Print out what the script is doing
echo "Installing htop"

# Install htop
make install

# Print out the completion message
echo "htop has been installed successfully to $INSTALL_DIR"
```

Export the path:

```bash
mkdir -p $HOME/modulefiles

cat << EOF > $HOME/modulefiles/htop
#%Module1.0
prepend-path PATH $INSTALL_DIR/bin
EOF

echo "module use --append $HOME/modulefiles" >> ~/.bashrc
source ~/.bashrc

echo "Module file for htop has been created at ~/modulefiles/htop"
```

Or if lmod isn't working on your PC(test witt ml avail) do the following:

```bash
echo "export PATH="$INSTALL_DIR/bin:$PATH"" >> ~/.bashrc
source ~/.bashrc
```
