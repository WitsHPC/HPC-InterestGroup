# Linux Command Line Tutorial Simulator

This project provides a guided terminal tutorial based on Ubuntu's "Command line for beginners" tutorial:

- Ubuntu guide: <https://ubuntu.com/tutorials/command-line-for-beginners#9-conclusion>

The application runs in a simulated Linux environment. Learners complete a sequence of levels covering navigation, file management, search, filtering, and simple pipelines without affecting the host filesystem.

Each session includes a maximum of three hints. The learner can reveal a hint with the `hint` command, and remaining hints are shown as hearts in the interface.

## Requirements

- Python 3.9 or newer recommended

Official Python downloads:

- <https://www.python.org/downloads/>

## Files

- `tutorial_app.py`

## Quick Start

Run the tutorial directly with Python:

```bash
python tutorial_app.py
```

If your system uses `python3` instead of `python`, run:

```bash
python3 tutorial_app.py
```

At the end of the tutorial, you should see:

- a `Run ID`
- a `Completion hash`

## Linux Guide

1. Open a terminal.
2. Change into the project directory:

```bash
cd /path/to/project
```

3. Run the tutorial:

```bash
python3 tutorial_app.py
```

If `python3` is not available but `python` is, use:

```bash
python tutorial_app.py
```

## Windows Guide

1. Install Python from the official Python downloads page: <https://www.python.org/downloads/>
2. During installation, enable the option to add Python to `PATH`.
3. Open Command Prompt or PowerShell.
4. Change into the project directory:

```powershell
cd C:\path\to\project
```

5. Run the tutorial:

```powershell
python tutorial_app.py
```

If `python` is not available, try:

```powershell
py tutorial_app.py
```

## In-App Commands

The simulator supports a focused set of commands:

- `pwd`
- `cd`
- `ls`
- `mkdir`
- `touch`
- `echo` with `>` and `>>`
- `cat`
- `cp`
- `mv`
- `rm -r`
- `grep`
- `find`
- simple `| wc -l` pipelines
- `help`
- `hint`
- `status`
- `clear`
- `exit`

## Notes

- The tutorial content and task progression are adapted from the Ubuntu command-line guide linked above.
