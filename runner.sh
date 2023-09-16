#!/bin/bash

docker start bias_bounty_security_container
docker start bias_bounty_repo_container

docker exec -d bias_bounty_repo_container sh -c "sh /home/($whoami)/actions-runner/_work/repo/repo/run.sh"