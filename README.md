# ECE765-511-Project
Repo for Team 511's project for ECE 765 - Probabilistic Graphical Models at NCSU.

Based off of research paper: https://www.cs.toronto.edu/~zemel/documents/cvpr04.pdf


Setup Steps:
	1) Create a conda environment and install at least the following packages for now:
		
		pytorch
		torchvision
		dataclasses
		

		- there currently is no requirements.txt, but perhaps we can make one
	
	2) From the command line run the install-packages.sh script by:
		
		`bash -i install-packages.sh`
		
		-i is needed for bash to automatically source your .bashrc file & enable conda commands within the script.
		
	3) From the command line run the download_data.sh script by:
	
		`bash -i download_data.sh`
		
		- This will clone the cityscapesScripts repo into the third-party folder (that will probably be created). You will need a .py file from there for downloading the data. This script will call that .py file and download the data to a folder that will be created called 'data' in the repo directory. After the data is downloaded, the script will call the cityscape_dataset.py file under data_prep/ to unzip the dataset into the appropriate file structure for utilizing the PyTorch.Datasets.CityScape utility.
		
		- After this successfully runs, you can delete the .zip files if you so desire.