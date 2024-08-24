---
title: Github Memo
date: 2024-04-02 14:40:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code_Utils, Toolbox]
math: True
---


## Create a Repo
1. Click the green button `New` on the GitHub repo website.
2. Do **not** check the `Add a README file`.
3. Copy the link with the `.git` extension.
4. Create a directory locally and enter it in a terminal.
5. Create a local git repo.
```bash
   git init
   ```
6. Link a local Git repository to a remote repository.
```bash
   git remote add origin ${xxx.git}
   ```

`${x}` means `x` is a variable.

The `origin` is a default name that refers to the original location (i.e., the remote repository's URL) from which you cloned the repository. When you use the `git clone ${URL}` command to clone a repository, Git automatically names the remote repository's URL as `origin`.

### Init Details
The `git init` command is used to create a new Git repository in the current directory. It initializes a hidden directory called `.git`, which contains all the metadata Git needs to track and manage version control for the project. Specifically, after executing `git init`, Git will set up the following:

- **HEAD**: A reference to the latest commit on the current branch.
- **index**: The staging area, where information about files that are about to be committed is recorded.
- **objects**: A directory that stores all the data, including files (blobs), directory trees (trees), and commit objects (commits).
- **refs**: A directory that stores pointers to commit objects, such as branches and tags.

### Remote Add Details

The command `git remote add origin ${xxx.git}` is used to link a local Git repository to a remote repository. Here's a breakdown of the command:

- `git remote add`: This part of the command tells Git that you want to add a new remote repository reference.
- `origin`: This is a conventional name used to refer to the primary or default remote repository. It's a shorthand alias that you can use to refer to the remote repository's URL in other Git commands.
- `${xxx.git}`: This is the placeholder for the remote repository's URL. You should replace `${xxx.git}` with the actual URL of your remote repository. The URL could be an HTTPS URL (e.g., `https://github.com/username/repository.git`) or an SSH URL (e.g., `git@github.com:username/repository.git`).

By executing this command, you're creating a connection between your local repository and a remote server. This connection allows you to push your commits from your local repository to the remote repository (using `git push`), as well as fetch changes from the remote repository to your local repository (using `git fetch` or `git pull`). This is essential for collaborating with others and for backing up your project on a remote server.

## Lazy Commit

Create a `snippet` in the software `Termius`:

```bash
git add ${files_name_you_want_to_commit}
git commit -m "${commit_info}"
git push origin ${branch_name}
```

Then enter your `github name` and your `git temporary token`.

Example:

```bash
git add .
git commit -m "quick commit"
git push origin main
```

Template:

1. Create a file `/bin/lazy_commit.sh`
2. ```bash
   #!/bin/bash
   echo "Input a commit message: "
   read commit_message
   git add .
   git commit -m "$commit_message"
   git push origin main
   ```
3. `chmod +x ./bin/lazy_commit.sh`
4. `./bin/lazy_commit.sh`

## .gitignore

Put a `${project_name}/.gitignore` file right under the project directory, then `git commit` will ignore the files listedin the `.gitignore` file.

Example: `.gitignore`

```
my_wandb_login_key.py
wandb/
.vscode/sftp.json
.DS_Store
```

It will ignore:
- The whole `${project_name}/wandb` directories.
- The `${project_name}/my_wandb_login_key.py` file.
- The `${project_name}/.vscode/sftp.json` file.
- All the files named `.DS_Store` in all directory under the project directory.

## Remove from Git

```bash
git rm -r --cached ${directory_name}
git rm --cached ${file_name}
```

## Download
1. Create a new terminal at the folder where you want to download the repo. The downloaded repo will be a subfolder, and its contents are what you see on the webpage.
2. `git clone ${repo_URL xxx.git}` (Download.) 
3. Enter the subfolder.

The `git clone` will create a subfolder (named after the repo) in your current folder.

## Branch

- `git branch -a` (List all the braches.)
- `git checkout ${branch_name}` (Switch to a branch.) 
- `git checkout -b ${branch_name}` (Create a branch.)

## Get Updated

### Way 1
1. `git fetch origin` (Retrieve the changes from all branches.)
2. `git merge origin/${remote_branch_name} ${local_branch_name}`


### Way 2
`pull` = `fetch` + `merge`

`git pull origin ${remote_branch_name}` (Update the code in your current local branch.)