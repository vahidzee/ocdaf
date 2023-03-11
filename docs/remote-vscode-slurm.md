# How to setup remote vscode on a slurm cluster

First off, you need to install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension. Then, you need to setup your ssh config file. This is usually located at `~/.ssh/config`. If you don't have one, you can create one. Here is an example of what it should look like:

```
    Host <name>
        HostName <hostname>
        User <username>
        IdentityFile <path to private key>
```

Using the remote-ssh you can simply develop on the remote server with all the extensions installed on the remote. In turn, you will have to setup your virtual environment and then either run a runnable python code, or a jupyter notebook.

## Setting up Jupyter notebook on a slurm cluster

Do do that, create a `templates` directory in your home directory. Then, create a `slrm` file in that directory with the following content

```
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -c 4
#SBATCH --mem=12GB
#SBATCH --output=notebook_output-%j.log

echo Running on $(hostname)
# env
source venv/bin/activate
# run
jupyter notebook --ip 0.0.0.0 --port 6858
```
This will create a jupyter notebook on port 6858 and you can use the `sbatch` command to set up that jupyter server on a specific device. For example, if your username is `johndoe` and the slurm file created above is in `~/templates/jupyter_template.slrm`, you can type in the following command:
```
sbatch ~/templates/jupyter_template.slrm
```
This will submit a job to the slurm network and will assign a node to the jupyter server that you can use. To check the status of the job, you can type in the following command:
```
squeue -u johndoe
```
This will show you the status of all the jobs that you have submitted. Once the job is running, you can use the following command to get the port number of the jupyter server:
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
8972746   rtx6000  jupyter   johndoe  R       0:14      1 gpu127
```
Now, you can use the jupyter extension of vscode to connect to the jupyter server. After openning the jupyter notebook, you can set the jupyter server to remote with the URI provided after the sbatch. Typically, the URI is of the form `http://hostname:6858/?token=xxxx`. After that, set up the virtual environment of the python being used and then you can start developing on the remote server with a better hardware!
