# MOPSbaseCode

This Readme is created for Student, Professors, Researchers and other interested parties for the project entitled “Molecular Odour Predictor Software”.
The original data has been placed in the folder “original Data”. The text files “molecular_descriptors_data.txt” and “TrainSet.txt” contain the input data and output data respectively. The “TrainSet.txt” file contains 21 labels, with the “INTENSITY” one being chosen for analysis as of right now [12-05-17].

There are three types of methodologies taken here, each stored in a different folder, named Methodology 1, 2 and 3.
The first one is the “Base Methodology”, or the creation of the relevant datasets. Here we use the methods of Scaling and PCA to create the different Datasets.
The second one is the running of the neural networks on the Base Data, Scaled Data and PCA Data.
The third one is the running of the neural network on the dropout variation on a randomly selected NN hidden layer size and the batch sizes.

The files in the second methodology follow the following pattern:
-	csvLogger: [training loss, learning rate, training metric, validation loss, validation metric] per epoch of training. Filenames: the filenames consist of the Hidden Layer Neuron number, Batch size, initial learning rate and default dropout rate. <No data on testing>
-	Report: contains the training and validation loss and correlation in graphs. The file names contain the type of graph, hidden layer neuron number, batch size, learning rate and dropout values. All images are in png format. <No data on testing>
-	Results: this folder contains the data related to the overall program run. This contains all the necessary details such as NN hidden layer size, batch size, learning rate, dropout rate, and the max, min and avg of the training and validation correlation and loss, as well as the testing Correlation and loss.
-	FINALRESULTS: this folder contains the results per step undergone. The folders here represent the data per step and the docx file contains the report generated for that particular run. Each Step folder contains the requisite information per step, such as the csvLogger, report and results folders of that particular step. A different folder name “visualizations” contain all the data and visualization for that particular step. The data however would have to be copied to that particular folder or the code changed in order to reflect the path of the folder. The graphics.py file contain the code for creating 2D graphs with the graphs3D.py containing the code for the 3D visualizations.

The dropout folder contains the data in the same format however without the FINALRESULTS folder since the data being presented itself is the final results. 

How to setup your virtual environment:
1.	Install Anaconda. Link: https://www.continuum.io/downloads. Install the Python 3.6 version according to your OS (Windows/Mac/Linux) and architecture (x32/x64)
2.	<Optional, but preferred>Create a virtual environment. CODE: CONDA CREATE -n “EnvName” python=3.5
3.	Type “PIP INSTALL -r req.txt” when in the code directory
4.	Add the code given in the next few lines to run the correlation metric

To run the codes, do either of the following:

Using Spyder IDE:
1.	Open spyder via virtual environment (if you have used one)
.		# activate “EnvName”
		#spyder
2.	Open the particular file you wish to run
3.	Run (or F5)

Using CMD:
1.	Go to the particular folder
2.	Start cmd via (Shift + right click > open command prompt)
3.	Run the code via: “python <FileName.py>”
