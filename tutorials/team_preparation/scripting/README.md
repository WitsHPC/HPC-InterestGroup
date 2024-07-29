# Scripting 
> Lily de Melo
> 

Scripting is where we put all our shell commands into a .sh file to helps us automate our workflows.

# Good practices when scripting: 

- Check if files or directories exists before downloading or `cd`’ing into them:

	```bash
	if [ -f "file_name" ]; then
		echo "file_name exists"
	else
		echo "file_name does not exist"
	fi
	```
	
	```bash
	mkdir -p your/diretory
	
	or
	
	[ -d your/directory ] && echo "Directory already exists" ||  mkdir -p your/directory
	```
  

- Remove old files before a new install for a clean build or after installing to clean up the directory:

	```bash
	echo "Removing old files"
	rm -rv $INSTALL_DIR/source_directory
	
	#download source_directory again or unzip it etc.
	```

- Error checking to stop your script when one command fails:

	```bash
	# exit when any command fails
	set -e
	
	# Handle exit and other signals
	trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?." 1>&2; fi' EXIT
	trap 'echo "Script interrupted" 1>&2; exit 2' INT TERM
	```

- Print out what your script is currently doing so that you know where your script failed/errored. Then you can go back and check the commands in that section:

	```bash
	echo "Loading dependencies"
	
	echo "Building"
	```

	- Using if-statements to specify what statements to print:

		```bash
		if make install; then
		    echo "Installation successful"
		else
		    echo "Installation failed"
		fi 
		```

- Use variables to keep paths consistent across your script:

	```bash
	#declaring variable
	INSTALL_DIR=/your/path
	
	#using variable
	mkdir -p $INSTALL_DIR
	```

	- Passing in variables through the bash command:

	  Script:
	    
	    ```bash
	    #!/bin/bash
	    
	    #declaring variable
	    INSTALL_DIR=$1
	    
	    #using variable
	    mkdir -p $INSTALL_DIR
	    ```
	    
	  Run command:
	    
	    ```bash
	    ./myscript.sh variable 
	    ```

- Setting environment variables in a script:

	```bash
	export PATH=/your/path:$PATH
	export LD_LIBRARY_PATH=/your/path/lib:$LD_LIBRARY_PATH
	
	#or use variables
	export PATH=$INSTALL_DIR:$PATH
	```

## Additional practices:

- Using sed to change values in a file:

	```
	sed [OPTIONS] 'COMMAND' [FILE]
	```

  - `s/pattern/replacement/`: Substitutes the first occurrence of the pattern with the replacement.
  - `s/pattern/replacement/g`: Substitutes all occurrences of the pattern with the replacement.
  - `d`: Deletes lines.
  - `i\text`: Inserts text before the pattern space.
  - `a\text`: Appends text after the pattern space.

	**Examples:**
	
	```bash
	cat > file.txt << 'EOF'
	old_thing
	old_thing
	old_thing
	EOF
	```
	
	```bash
	sed -i 's/old_thing/new_thing/g' file.txt #changes all the lines
	
	sed -i '/new_thing/i\this_is_' file.txt
	
	sed -i '/new_thing/a\_is_here' file.txt
	```

- Creating your own file instead of `sed`. Useful for input files:

	```
	cat > newfile << 'EOF'
	This is the first line of my new file.
	This is the second line of my new file.
	EOF
	```

  - You can create lmod modules within your script so that you don’t need to change the module file every time you change your script.

	   **Examples:**
	    
	   ```bash
	    cat << EOF > $INSTALL_DIR/modules/openmpi
	    #%Module1.0
	    prepend-path PATH $INSTALL_DIR/bin
	    prepend-path LD_LIBRARY_PATH $INSTALL_DIR/lib
	    EOF
	   ```
    Or
	    
	 ```bash
	    cat <<EOF > "$MODULES_DIR/openmpi.lua"
	    -- openmpi module file
	    help([[OpenMPI is a High Performance Message Passing Library.]])
	    
	    whatis("Name: OpenMPI")
	    whatis("Version: 1.2.11")
	    whatis("Keywords: mpi, parallel, openmpi")
	    
	    -- Specify the paths
	    local base = "$INSTALL_DIR"
	    
	    prepend_path("PATH", pathJoin(base, "bin"))
	    prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib"))
	    prepend_path("CPATH", pathJoin(base, "include"))
	    EOF
	 ```

# Advantages of scripting:

- Your work remains consistent. Each time you run an app, the process will be the same, so you don't need to remember all the steps. This consistency makes debugging quicker since you know exactly what you've done.

- You can replicate work across different systems. A script you create on your laptop can be used on a cluster if made correctly.

- Automating workflows saves time. For instance, you can have a script for installing btop or Intel tools. This will save you time in the future by avoiding repetitive manual steps.

# Tutorial

Make a script to download and install btop. Add all the bells and whistles of path checking, exporting etc.

Here are the basic steps for installing btop:

```
git clone --recursive https://github.com/aristocratos/btop.git
```
```
cd btop
```

```jsx
mak
```

```jsx
make install PREFIX=/your/path
```

```jsx
export PATH=$PATH:/your/path
```
