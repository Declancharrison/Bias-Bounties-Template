#!/bin/bash

if [ "$(docker ps -a -q -f name=bias_bounty_container)" ]; then
      # cleanup
      echo "Removing Container"
fi