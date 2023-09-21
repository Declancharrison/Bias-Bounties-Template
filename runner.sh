#!/bin/bash

docker start bias_bounty_security_container
docker start bias_bounty_repo_container

SHELL=/bin/bash xterm -e 'docker exec -it bias_bounty_security_container bash -c "python3 ~/server.py"' & 

docker exec -it bias_bounty_repo_container bash -c ". ~/repo/actions-runner/run.sh"