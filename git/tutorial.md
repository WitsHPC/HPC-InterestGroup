## Git Basics and Cool Git Features Tutorial

### Introduction

Git is a powerful distributed version control system that allows you to track changes in your code, collaborate with others, and manage your projects efficiently. In this tutorial, we will cover the basics of Git and introduce you to some cool Git features using sample files to illustrate each concept.

### Prerequisites

Before we dive into Git, make sure you have the following prerequisites installed on your computer:

1. **Git**: Download and install Git from [https://git-scm.com/](https://git-scm.com/).

2. **Code Editor**: Use a code editor of your choice. Popular options include Visual Studio Code, Atom, or Sublime Text.

### Git Basics

#### Initializing a Git Repository

1. Open your terminal or command prompt.

2. Navigate to your project folder. For example:

   ```bash
   cd ~/projects/my_project
   ```

3. To initialize a new Git repository, run the following command:

   ```bash
   git init
   ```

#### Cloning a Repository

To work with an existing Git repository hosted on a platform like GitHub or GitLab, use the `git clone` command. Let's clone a sample repository:

```bash
git clone https://github.com/yourusername/sample-repo.git
```

#### Staging and Committing Changes

1. Make changes to your project files. Create two sample files: `file1.txt` and `file2.txt`.

2. To stage changes for commit, use the `git add` command:

   ```bash
   git add file1.txt file2.txt
   ```

3. Commit the staged changes with a meaningful message:

   ```bash
   git commit -m "Added file1.txt and file2.txt"
   ```

#### Checking Status and History

- Check the status of your repository with:

  ```bash
  git status
  ```

- View the commit history:

  ```bash
  git log
  ```

#### Branching and Merging

1. Create a new branch and switch to it:

   ```bash
   git checkout -b feature-branch
   ```

2. Make changes to `file1.txt`.

3. Commit the changes:

   ```bash
   git add file1.txt
   git commit -m "Updated file1.txt in feature branch"
   ```

4. Switch back to the main branch:

   ```bash
   git checkout main
   ```

5. Merge changes from the feature branch into the main branch:

   ```bash
   git merge feature-branch
   ```

### Cool Git Features

Now, let's explore some cool Git features using the sample files:

#### 1. Git Stash

The `git stash` command allows you to temporarily save changes that are not ready to be committed. This is useful when you need to switch branches or perform other operations without committing your current changes.

- Make changes to `file1.txt`.

- To stash changes:

  ```bash
  git stash
  ```

- Make more changes to `file2.txt`.

- Apply the stashed changes:

  ```bash
  git stash apply
  ```

#### 2. Git Interactive Rebase

Interactive rebase allows you to edit, reorder, or squash commits before pushing them to the remote repository.

- Start an interactive rebase to edit the last two commits:

  ```bash
  git rebase -i HEAD~2
  ```

- In the text editor that opens, change `pick` to `edit` for the second commit.

- Save and close the editor.

- Amend the commit:

  ```bash
  git commit --amend
  ```

- Continue the rebase:

  ```bash
  git rebase --continue
  ```

#### 3. Git Bisect

Git bisect helps you find the commit that introduced a bug by using binary search.

- Start a bisect session:

  ```bash
  git bisect start
  ```

- Mark a commit as good or bad:

  ```bash
  git bisect good       # If a commit is good
  git bisect bad        # If a commit is bad (bug introduced)
  ```

- Git will guide you to the commit that introduced the bug.

#### 4. Git Hooks

Git hooks allow you to automate actions at specific points in the Git workflow, such as before or after a commit.

- Explore the `.git/hooks` directory in your repository to set up custom hooks.

#### 5. Git Actions

You can setup actions for when code gets pushed to a repository on GitHub

Read more here: https://docs.github.com/en/actions/quickstart

### Conclusion

Congratulations! You've now learned Git basics and explored some cool Git features using sample files. Git is a powerful tool for version control and collaboration in software development. Continue to practice and experiment with these features to become a Git expert. Happy coding!
