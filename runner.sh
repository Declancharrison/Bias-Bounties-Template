#!/bin/bash

docker start bias_bounty_container

path="$(pwd)"
IFS='/' read -ra REPO_NAME <<< "${path}"
echo "$path/actions-runner/_work/${REPO_NAME[-1]}/${REPO_NAME[-1]}/run.sh"