#!/bin/sh

# Install CI dependencies, arguments specify what is required

# Base packages
pip install --upgrade pip
pip install wheel setuptools

# Used to create local test environments
pip install tox

# Update apt
sudo apt-get update

# Check arguments
for par in "$@"; do
  case $par in
    doc)
      # documentation
      sudo apt-get install pandoc
    ;;

    *)
      echo "Unknown argument: $par" >&2
      exit 1
    ;;
  esac
done
