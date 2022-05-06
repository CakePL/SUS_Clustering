#!/bin/bash
VENV="virtualenv venv"
VENV_ACTIVATE="source venv/bin/activate"
REQ="pip install -r requirements.txt"
PREINFO="To install clustering application make sure You have installed following prerequisites:
    * python3
    * python3-virtualenv
    * python3-pip

While installing some data can be used to satisfy application requirements.
"
POSTINFO="To run application launch \"python3 cluster.py <filename>\".
<filename> have to be txt with paths to images, which you want to cluster.
You can also launch app with randomly generated paths to test data,
using \"python3 cluster.py random\"."

echo -------------------------- SUS Clustering --- INSTALLATION --------------------------
printf "$PREINFO"
read -p "Do you want to install this program? [y/n] " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo Aborting installation.
    exit 1
fi

printf "Beginning installation process.\n"
printf "Creating virtual environment...\n"
eval "$VENV"
eval "$VENV_ACTIVATE"
printf "\nVenv created and activated!\n"

printf "Installing requirements...\n"
eval "$REQ"
printf "\nRequirements installed!\n\n"

printf "$POSTINFO"

printf "Installation completed!\n\n"
