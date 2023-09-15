#!/bin/bash

echo "\n\
      +===============================================+ \n\
      +                                               + \n\
      +      Welcome to the installation wizard!      + \n\
      +                                               + \n\
      +===============================================+\n"

echo "Building python virtual environment"
pyc="$(python3 -V)"
pya=( $pyc )
IFS='.' read -ra VER <<< "${pya[1]}"
sudo apt install -y "python${VER[0]}.${VER[1]}-venv"

cp -a installation/. .

rm -R installation/

python3 -m venv bias_bounty_venv

sudo chmod -R +rwx bias_bounty_venv/

sudo $(pwd)/bias_bounty_venv/bin/pip install -r requirements.txt

python3 setup.py

ret=$?
if [ $ret -ne 0 ]; then
     echo "Installation Error, please see error messages and retry!"
     exit
fi
echo "Testing Docker"

if ! command -v docker &> /dev/null; then
      echo "Docker not found, trying installation"
      sudo apt-get update
      sudo apt-get install ca-certificates curl gnupg
      sudo install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      sudo chmod a+r /etc/apt/keyrings/docker.gpg

      # Add the repository to Apt sources:
      echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      sudo apt-get update
fi


echo "Creating docker group + Adding User"
sudo groupadd docker
sudo usermod -aG docker $(whoami)
sudo chown :2021 -R $(pwd)
sudo chmod 755 $(pwd)
sudo chmod g+s $(pwd)
adduser $(whoami) 2021

echo "Building Docker Security Container"

docker build -f Dockerfile.sec -t bias_bounty_security:1.0 .
if [ "$(docker ps -a -q -f name=bias_bounty_security_container)" ]; then
      # cleanup
      echo "Removing Container"
      docker rm bias_bounty_security_container
fi
echo "Creating Container"
docker run -v $(pwd)/container_tmp:/home/non-root/container_tmp --network none  --name bias_bounty_security_container -it bias_bounty:1.0

exit

SECRET_USER=$(python3 -c "import setup; print(setup.create_env())")
docker build --build-arg SECRET_USER="$SECRET_USER" -f Dockerfile.repo -t bias_bounty_repo:1.0 .
if [ "$(docker ps -a -q -f name=bias_bounty_repo_container)" ]; then
      # cleanup
      echo "Removing Container"
      docker rm bias_bounty_repo_container
fi
docker run -v $(pwd):/home/$SECRET_USER/repo --name bias_bounty_repo_container -it bias_bounty_repo:1.0

exit

echo "Installation Complete!"
# rm Dockerfile
# rm bad_argvals.txt
