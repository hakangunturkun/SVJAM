#!/bin/bash 

# This is an example script. Please provide the respective file location for loupe server and loupe files 
# https://support.10xgenomics.com/genome-exome/software/visualization/latest/installation 


LOUPE_INSTALLATION_DIR='/home/'

cd $LOUPE_INSTALLATION_DIR

# put .loupe files in the dir below 
export LOUPE_SERVER="/mnt/media/mouse_loupe_files"

./start_loupe.sh
