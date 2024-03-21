# Extra Tut Steps

Below are the steps to complete the Extra Tutorial.

1. **Download the `extra-tut-shell.tar.gz` file.**
   - This tutorial assumes that it is in your user home directory `cd ~`. If not, move the file:
```bash
cd /path/to/file
mv extra-tut-shell.tar.gz ~
```

2. **Create a Directory:**
   - Run the following command in your terminal to create a directory named `temp-hpc`:

```bash
mkdir temp-hpc
```

3. **Extract Files:**
   - Extract the contents of the `temp_contents.tar.gz` archive (download from moodle) into the `temp-hpc` directory. Run the following command:

```bash
tar -xzvf extra-tut-shell.tar.gz -C ~/temp-hpc/
```

4. **Navigate to the Directory:**
   - Change your current directory to `temp-hpc`:

```bash
cd temp-hpc
```

5. **Search for Errors:**
   - Use the `grep` command to search for instances of the word "ERROR" in the log files within the `logs` directory:

```bash
cd logs
grep -r "ERROR"
```

6. **Find the Number:**
   - After running the previous command, you'll find an error that does not look like the others. Make a note of this code. **XXXX XX XXX XXXX XX XXXXX**

7. **Update the Message:**
   - Change your directory to `src`:

```bash
cd ../src
```

   - Open the `message.txt` file and put the code you found in step 5 into it.


8. **Create the Run Script:**
   - Create a run script named `run.sh` with the following content:

```
vim run.sh
```

(remember to use **insert** and **esc + :wq** - press them as buttons, do not hold down) 


```bash
#!/bin/bash -e

 # Check if the main.py script exists
if [ ! -f "main.py" ]; then
	echo "Error: main.py not found"
    exit 1
fi

# Run the Python script
python3 main.py

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "Python script executed successfully"
else
    echo "Error: Python script execution failed"
    exit 1
fi
```  

9. **Run the Python Script:**
   - Execute the Python script to decode the message. Run the following command:

```bash
bash ./run.sh
```
- You may have to use ```chmod +x``` to change the file permissions.
