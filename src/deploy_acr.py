import os
from dotenv import load_dotenv
import paramiko

from utils import Utils
import json

class DeployAzure:
    def __init__(self, ssh: paramiko.SSHClient):
        load_dotenv()
        self.config = Utils().load_config()
        self.ssh = ssh
        self.LINUX_HOST = os.getenv("LINUX_HOST")
        self.LINUX_USERNAME = os.getenv("LINUX_USERNAME")
        self.LINUX_PASSWORD = os.getenv("LINUX_PASSWORD")
        self.REMOTE_DIR = os.getenv("REMOTE_DIR")
        self.CLIENT_ID = os.getenv("CLIENT_ID")
        self.CLIENT_SECRET = os.getenv("CLIENT_SECRET")
        self.TENANT_ID = os.getenv("TENANT_ID")
        self.OBJECT_ID = os.getenv("OBJECT_ID")
        self.ACR_NAME = os.getenv("ACR_NAME")
        self.RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")

    def deploy_acr(self):
        """Deploy the Docker image to Azure Container Instances (ACI).

        This method assumes that the Docker image has been built and is available
        on the remote Linux host. It uses Azure CLI commands to create a resource
        group, create a container registry, push the Docker image to the registry,
        and then deploy the container instance in Azure.

        Note: This is a simplified example. In a production scenario, you would
        need to handle authentication, error checking, and cleanup of resources.
        """
        # 1. Login to Azure (On Linux)
        login_cmd = f'''az login --service-principal --username {self.CLIENT_ID} --password '{self.CLIENT_SECRET}' --tenant {self.TENANT_ID}'''
        
        stdin, stdout, stderr = self.ssh.exec_command(login_cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        print("az login stdout:", out)
        if err:
          print("az login stderr:", err)
        if code != 0:
          raise RuntimeError(f"az login failed with exit code {code}")

        print("Logged in to Azure successfully.")

        # 2. Login to ACR
        acr_login_cmd = f'''az acr login --name {self.ACR_NAME}'''
        stdin, stdout, stderr = self.ssh.exec_command(acr_login_cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        print("az acr login stdout:", out)
        if err:
          print("az acr login stderr:", err)
        if code != 0:
          raise RuntimeError(f"az acr login failed with exit code {code}")

        print("Logged in to Azure Container Registry successfully.")

        # 3. Tag the Docker image for ACR
        tag_cmd = f'''docker tag {self.config["docker_image_name"]}:latest {self.ACR_NAME}.azurecr.io/{self.config["docker_image_name"]}:latest'''

        stdin, stdout, stderr = self.ssh.exec_command(tag_cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        print("docker tag stdout:", out)
        if err:
          print("docker tag stderr:", err)
        if code != 0:
          raise RuntimeError(f"docker tag failed with exit code {code}")

        print("Docker image tagged for Azure Container Registry successfully.")

        #4. Push the Docker image to ACR
        push_cmd = f'''docker push {self.ACR_NAME}.azurecr.io/{self.config["docker_image_name"]}:latest'''

        stdin, stdout, stderr = self.ssh.exec_command(push_cmd)
        # Stream output may be large; read fully for debugging
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        print("docker push stdout:\n", out)
        if err:
          print("docker push stderr:\n", err)
        if code != 0:
          raise RuntimeError(f"docker push failed with exit code {code}")

        print("Docker image pushed to Azure Container Registry successfully.")

        # 5. Verify repository exists in ACR (helpful for portal troubleshooting)
        verify_cmd = f"az acr repository list -n {self.ACR_NAME} --output json"
        stdin, stdout, stderr = self.ssh.exec_command(verify_cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        print("az acr repository list stdout:\n", out)
        if err:
          print("az acr repository list stderr:\n", err)
        if code != 0:
          print(f"Warning: 'az acr repository list' returned exit code {code} (may indicate permissions or CLI issue)")

        # Also show tags for the specific repository to confirm the pushed tag
        repo_name = self.config.get("docker_image_name")
        if repo_name:
          tags_cmd = f"az acr repository show-tags -n {self.ACR_NAME} --repository {repo_name} --output json"
          stdin, stdout, stderr = self.ssh.exec_command(tags_cmd)
          tout = stdout.read().decode('utf-8')
          terr = stderr.read().decode('utf-8')
          tcode = stdout.channel.recv_exit_status()
          print(f"az acr repository show-tags stdout for {repo_name}:\n", tout)
          if terr:
            print(f"az acr repository show-tags stderr:\n", terr)
          if tcode != 0:
            print(f"Warning: 'az acr repository show-tags' returned exit code {tcode}")


    def deploy_aci(self):
        #1. Deploy the container instance in Azure
        deploy_cmd = (
            f"az container create "
            f"--resource-group {self.RESOURCE_GROUP} "
            f"--name {self.ACR_NAME} "
            f"--image {self.ACR_NAME}.azurecr.io/{self.config['docker_image_name']}:latest "
            f"--cpu 1 --memory 2 --ports 8100 --ip-address Public "
            f"--os-type Linux "
            f"--registry-login-server {self.ACR_NAME}.azurecr.io "
            f"--registry-username {self.CLIENT_ID} "
            f"--registry-password '{self.CLIENT_SECRET}'"
        )

        print(f"Deploying {self.config['docker_image_name']}...")
        stdin, stdout, stderr = self.ssh.exec_command(deploy_cmd)
        tout = stdout.read().decode('utf-8')
        terr = stderr.read().decode('utf-8')
        tcode = stdout.channel.recv_exit_status()
        print(f"Output for Deploying ACI command:\n", tout)
        
        if terr:
            print(f"Error deploying ACI:\n", terr)
            raise RuntimeError(f"Deploying ACI failed with exit code {tcode}")
        if tcode != 0:
            print(f"Warning: Deploying ACI failed with exit code {tcode}")
            raise RuntimeError(f"Deploying ACI failed with exit code {tcode}")
        # print("Container instance deployed in Azure successfully.")

        #2. Get ACI public IP address
        ip_cmd = f'''az container show\
          --resource-group {self.RESOURCE_GROUP}\
          --name {self.ACR_NAME}\
          --query ipAddress.ip\
          --output tsv'''
        stdin, stdout, stderr = self.ssh.exec_command(ip_cmd)
        tout = stdout.read().decode('utf-8')
        terr = stderr.read().decode('utf-8')
        tcode = stdout.channel.recv_exit_status()
        print(f"Output for getting IP:\n", tout)
        
        if terr:
            print(f"Error getting IP:\n", terr)
            raise RuntimeError(f"Getting IP failed with exit code {tcode}")
        if tcode != 0:
            print(f"Warning: Getting IP failed with exit code {tcode}")
            raise RuntimeError(f"Getting IP failed with exit code {tcode}")
        
        # # Read outputs from the buffers
        # ip = stdout.read().decode('utf-8')
        # error = stderr.read().decode('utf-8')
        # print(f"Retrieved ACI public IP address successfully: {ip}.")
        
        # #3. Test the deployed service using curl
        # test_cmd = f'''curl -X GET http://{ip}:{os.getenv("CONTAINER_PORT")}/health'''
        # stdin, stdout, stderr = self.ssh.exec_command(test_cmd)
        # output = stdout.read().decode('utf-8')
        # error = stderr.read().decode('utf-8')

        # print(output)