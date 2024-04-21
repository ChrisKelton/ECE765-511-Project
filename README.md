# ECE765-511-Project
Repo for Team 511's project for ECE 765 - Probabilistic Graphical Models at NCSU.

Based off of research paper: https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf


Setup Steps:
	
	1) Create a conda environment and utilize the requirements.txt for necessary packages.
	
	2) From the command line run the install-packages.sh script by:
		
		`bash -i install-packages.sh`
		
		-i is needed for bash to automatically source your .bashrc file & enable conda commands within the script.
		
	3) From the command line run the download_data.sh script by:
	
		`bash -i download_data.sh`
		
		- This will clone the cityscapesScripts repo into the third-party folder (that will probably be created).
			You will need a .py file from there for downloading the data. This script will call that .py file and 
			download the data to a folder that will be created called 'data' in the repo directory. 
			After the data is downloaded, the script will call the cityscape_dataset.py file under data_prep/ to 
			unzip the dataset into the appropriate file structure for utilizing the PyTorch.Datasets.CityScape utility.
		
		- After this successfully runs, you can delete the .zip files if you so desire.


Training HyperParameters:

    - batch_size = 1
    - learning_rate = 1e-5
    - weight_decay = 3e-4
    - optimizer = AdamW
    - loss = CrossEntropy Loss
    - epochs = 50
    - epochs_to_test_val = 2
    
    batch size is set to 1 for computational resources, since the images are ~1024x2048 each.
    Since this would result in high variation in the variance of the gradient, we set the learning
    rate very low (as in the paper linked above). This proved to help in training substantially.
    As, previously, we experimented with using the center crop of the images at size 624x624 w/ a learning rate of 
    1e-3 and the Adam optimizer, which resulted in a validation accuracy ~60% after 10 epochs; resizing the images 
    to 224x224 w/ a learning rate of 1e-3 and the Adam optimizer, which resulted in a validation accuracy oscillating
    around ~50% for 50 epochs; not resizing or cropping the images and just using a batch size of 1, learning rate 
    of 1e-5 and AdamW optimizer, which resulted in a validation accuracy 84%+.


Results:

| Model                                                                | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss | Epochs |
|:---------------------------------------------------------------------|:-----------------:|:-------------------:|:-------------:|:---------------:|:------:|
| FcnResNet (frozen backbone layers)                                   |      87.54%       |       86.33%        |     0.394     |      0.442      |   30   |
| CRF-RNN w/ FcnResNet Backbone (frozen backbone layers)               |      87.63%       |       86.54%        |     0.387     |      0.432      |   30   |
| CRF-RNN w/ FineTuned FcnResNet Backbone (frozen backbone layers)     |      89.12%       |       87.01%        |     0.336     |      0.422      |   30   |
| CRF-RNN w/ FineTuned FcnResNet Backbone (non frozen backbone layers) |                   |                     |               |                 |        |
