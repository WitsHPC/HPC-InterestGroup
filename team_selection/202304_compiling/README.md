# Assignment: Compiling btop from Source
By: Sayfullah Jumoorty, with original instructions [here](https://github.com/aristocratos/btop#compilation-linux).
## Introduction
In this assignment, you will learn how to compile btop, a task manager and resource monitoring tool, from source on your own machine. Compiling from source allows you to customize the build and have the latest version of the software. You will follow the instructions provided in the [btop GitHub repository](https://github.com/aristocratos/btop#compilation-linux) to complete the task.

## Prerequisites
Before you begin, ensure that you have the following prerequisites set up on your machine:
- Linux operating system (recommended: Ubuntu)
- Git installed
- Development tools and libraries (build-essential, libncurses-dev)
  - *Lab machines should have this. If not, install from source*. [Here](https://github.com/mirror/ncurses/blob/master/INSTALL) and [here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/team_selection/202304_compiling/ncurses.md) are instructions for ncurses.

## Instructions
Follow the steps below to compile btop from source on your machine:

1. **Clone the btop repository:** Open a terminal and navigate to a suitable location on your machine. Then, execute the following command:
   ```bash
   git clone --recursive https://github.com/aristocratos/btop.git
   ```

2. **Change to the btop directory:** Move into the newly created `btop` directory using the following command:
   ```bash
   cd btop
   ```

3. **Build btop:** Run the following command to initiate the build process:
   ```bash
   make
   ```

4. **Install btop:** After the build process completes successfully, install btop using the following command:
   ```bash
   make install PREFIX=/your/install/directory/choice/here
   ```

6. **Export build path:** Add the path of the build_directory/bin into your ~/.bashrc file:
   ```bash
   vim ~/.bashrc --> then add the export inside --> then source your bashrc file
   ```

6. **Verify installation:** Verify that btop has been installed correctly by running the command:
   ```bash
   btop
   ```

   This should launch the btop application, and you should be able to see the resource monitoring interface.

## Submission
Once you have successfully compiled and installed btop, take a screenshot of the running btop application and write a brief reflection on the experience of compiling from source.

Submit the following materials as your assignment:
1. The screenshot of the running btop application.
2. A brief reflection on your experience of compiling btop from source. Include any challenges you faced, what you learned from the process, and your overall impression.

## Conclusion
Congratulations on completing the assignment! By compiling btop from source, you have gained hands-on experience with building software and installing it on your machine. This process allows you to customize the software to suit your needs and keep it up to date with the latest features and improvements. Understanding the compilation process is an essential skill for any developer, and it opens up opportunities to contribute to open-source projects or create your own software in the future. Keep exploring and experimenting with different projects to deepen your knowledge in software development.