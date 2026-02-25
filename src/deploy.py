"""Deployment orchestration for containerized model serving.

This module provides `Deploy`, a class that automates the process of:
- Transferring project files to a remote Linux host via SSH/PSCP
- Building a Docker image from a Dockerfile
- Running the containerized application
- Testing the deployed service
- Promoting to cloud platforms (Azure Container Instances)

Configuration is loaded from environment variables (.env file).
"""

from dotenv import load_dotenv
import os
import subprocess
from utils import Utils
import paramiko
from deploy_acr import DeployAzure


class Deploy:
    """Orchestrates deployment of the churn prediction model to remote infrastructure.

    This class handles SSH connections, file transfers, Docker image building,
    container orchestration, and service testing on a remote Linux host.

    Environment Variables (from .env):
        LINUX_HOST: IP/hostname of the remote Linux server
        LINUX_USER: SSH username
        LINUX_PASSWORD: SSH password
        REMOTE_DIR: Directory on remote host for deployment
        WINDOWS_DIR: Local directory to transfer from
        LINUX_USERNAME: Username for SCP/Docker registry
    """

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        # Load application configuration
        self.config = Utils().load_config()

        # SSH and deployment credentials from environment
        self.LINUX_HOST = os.getenv("LINUX_HOST")
        self.LINUX_USER = os.getenv("LINUX_USER")
        self.LINUX_PASSWORD = os.getenv("LINUX_PASSWORD")
        self.REMOTE_DIR = os.getenv("REMOTE_DIR")

    def create_ssh_client(self) -> paramiko.SSHClient:
        """Establish an SSH connection to the remote Linux host.

        Returns:
            paramiko.SSHClient: Connected SSH client object

        Prints:
            Connection status messages
        """
        print(f"Connecting to {self.LINUX_HOST}...")
        ssh = paramiko.SSHClient()
        # Automatically accept unknown hosts (use with caution in production)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            self.LINUX_HOST,
            username=self.LINUX_USER,
            password=self.LINUX_PASSWORD,
            look_for_keys=False,  # Don't search ~/.ssh/ for keys
            allow_agent=False,  # Don't use SSH agent
        )
        print(f"sucessfully connected to {self.LINUX_HOST}")
        return ssh
    
    def transfer_to_linux(self, ssh: paramiko.SSHClient) -> None:
        """Transfer project files to the remote Linux host using SCP.

        Creates the remote directory and recursively transfers all project
        files (source code, config, data, Docker resources, etc.) to enable
        remote deployment.

        Args:
            ssh (paramiko.SSHClient): Connected SSH client

        Prints:
            Transfer status for each file/directory
        """
        # Ensure remote directory exists
        ssh.exec_command(f"mkdir -p {self.REMOTE_DIR}")

        # List of files and directories to transfer
        files = [
            "src",
            "config.yaml",
            "requirements.txt",
            "data",
            "Dockerfile",
            ".env",
            ".gitignore",
            "test.py",
            "README.md",
            "mlflow.db",
            "exported_model"
        ]

        for i in files:
            # Build SCP command to transfer files with password authentication
            command = [
                "pscp",
                "-pw",
                self.LINUX_PASSWORD,
                "-r",  # Recursive copy for directories
                os.path.join(os.getenv("WINDOWS_DIR"), i),
                f"{os.getenv('LINUX_USERNAME')}@{os.getenv('LINUX_HOST')}:{os.getenv('REMOTE_DIR')}",
            ]
            try:
                # Execute the SCP transfer command
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Transfer successful:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Transfer failed:", e.stderr)
    
    def create_docker_image(self, ssh: paramiko.SSHClient, image_name: str)-> None:
        """Build a Docker image from the Dockerfile on the remote host.

        Args:
            ssh (paramiko.SSHClient): Connected SSH client
            image_name (str): Name/tag for the Docker image (e.g., 'user/churnprediction')
        """
        # Build Docker image from Dockerfile in the remote directory
        docker_command = f"docker build -t {image_name} {self.REMOTE_DIR}"
        stdin, stdout, stderr = ssh.exec_command(docker_command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        print(f"Output: {output}")
        print(f"Error: {error}")
    
    def run_docker_container(self, ssh: paramiko.SSHClient, image_name: str)-> None:
        """Run a Docker container from the built image on the remote host.

        Args:
            ssh (paramiko.SSHClient): Connected SSH client
            image_name (str): Name/tag of the Docker image to run
        """
        # Run container with port mapping (container:8100 -> host:8103)
        docker_command = f"docker run -p {os.getenv('CONTAINER_PORT')}:8100 {image_name}"
        stdin, stdout, stderr = ssh.exec_command(docker_command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        print(f"Output: {output}")
        print(f"Error: {error}")

    def test_docker(self, ssh: paramiko.SSHClient, image_name: str)-> tuple:
        """Execute tests on the deployed service to validate functionality.

        Runs `test.py` on the remote host and captures stdout, stderr, and
        exit code to determine if the deployment succeeded.

        Args:
            ssh (paramiko.SSHClient): Connected SSH client
            image_name (str): Docker image name (included for consistency)

        Returns:
            tuple: (status_str, output_str) where status is 'SUCCESS' or 'FAILED'
        """
        command = f"python3 {self.REMOTE_DIR}/test.py"
        stdin, stdout, stderr = ssh.exec_command(command)

        # Read outputs from the buffers
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        # Get the exit code (0 = success, non-zero = failure)
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            print("Command successful!")
            return "SUCCESS", output
        else:
            print(f"Command failed with status {exit_status} and Error: {error}")
            return "FAILED", output
    
    def __call__(self):
        """Orchestrate the full deployment pipeline.

        Coordinates SSH connection, file transfer, Docker image building,
        container execution, and service testing. Provides status feedback
        and guides next steps (e.g., Azure deployment).
        """
        print("==========================================")
        print("STARTING SECURE DEPLOYMENT PIPELINE")
        print("==========================================")

        image_name = self.config["docker_image_name"]
        # Establish SSH connection to remote host
        ssh = self.create_ssh_client()
        # Transfer all project files to remote host
        print(f"Transfering files to linux host: \n{self.transfer_to_linux(ssh)}")

        # Build Docker image on remote host
        print(f"Creating docker image: \n{self.create_docker_image(ssh, image_name)}")

        # Start Docker container with port mapping
        print(f"Running docker image: \n{self.run_docker_container(ssh, image_name)}")

        # Run validation tests on the deployed service
        status, output = self.test_docker(ssh, image_name)
        print(f"Testing docker image: \n{status}, {output}")

        # Provide next steps based on test results
        if status == "SUCCESS":
            print("Docker worked perfectly. Deployng it in Azure Container Instance")
            da = DeployAzure(ssh)
            # try:
            da.deploy_acr()
            # except Exception as e:
            #     raise f"Error in deploying acr: {e}"
            
            # try:
            da.deploy_aci()
            # except Exception as e:
            #     raise f"Error in deploying aci: {e}"
        else:
            print("Docker Image not working properly. Please check the logs and fix the issue.")


# Initialize and run deployment pipeline
deploy = Deploy()
deploy()