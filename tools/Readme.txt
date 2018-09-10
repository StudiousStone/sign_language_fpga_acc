This is a bunch of tools for the generation of test benches for our neural network.

Right now, the information in goldendata folder has been generated using "high_accuracy_and_loss"
model and classifying the image A_1.jpg. To edit those details, check the current file and
directory names at the beginning of function gen_test_data() in file classify.py

All the data is stored following the next pattern, from outside to inside:

--layer,ch_out,ch_in,row,col

If some of them are non existent, they are ignored
e.g: bias only has layer and ch_out. First channels from first layer, then channels from second layer, etc.
	 file conv1 only has ch_out,row,col.

Inputs should be removed, paths fixed, template.h adjusted and val_data populated.
Could also check the models folder
