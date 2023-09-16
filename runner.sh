#!/bin/bash

docker start bias_bounty_security_container
docker start bias_bounty_repo_container

docker exec -it bias_bounty_repo_container bash -c ". ~/repo/actions-runner/run.sh"