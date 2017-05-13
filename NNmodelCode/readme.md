This folder contains 4 files:
- runStep1.py
- runStep2.py
- runStep3.py
- runVaryDrop.py

The first three may be explained in detail via the Project Report generated for the BTech Training of Arpit Mathur (RSDC6991 GitHub handle)
alongwith an explaination to the NN aproach itself.
Link1: https://goo.gl/fy8xiJ
Link2: https://goo.gl/EY6Vm8

runStep1:    computes multiple models with variation on NN hidden layer size
runStep2:    computes multiple models with variation on NN batch size
runStep3:    computes multiple models with variation on NN dropout rate
runVaryDrop: computes a model with variation on dropout rates on a randomly selected NN hidden layer size and batch size.

For each dataset, do the following steps:
Step 1: Create Script to scale data from original
Step 2: Run Neural Networks with variations in:
-	Number of Neurons in Hidden Layer*
Step 3: Analyze Results
Step 4: Take optimal Neural Network (s) and run Neural Network with Variations in:
-	Batch Size*
Step 5: Analyze Results
Step 6: Take optimal Neural Network (s) and run Neural Network with Variations in:
-	Dropout Rates*
Step 7: Analyze Results

*Ranges:
1: Run varies between 15 and 400
2: Run varies between 1 to 68
3. Learning Rates vary in [0.01, 0.02, 0.05, 0.1, 0.15,0.20,0.25]
