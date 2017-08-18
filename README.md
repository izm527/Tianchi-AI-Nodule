1. making_mask.py  data_preprocess.ipynb: 
	use to do data-prepocessing and removing the unrelated vessel organs from the meta data. 
	Output the scans only containing nodules.


2. gen_unet_inputs.py: 
	Use to partly cut the scan space containing the nodules(ll-----hh) from the prepossed 
	scan(as input data) and masked scan(as labels).

3. unet_training.py  Unet-train.ipynb:
	Use the data and labels from step 2 to train a UNET network for nodule segment!

4. gen_detection_data.py:
	Partly clip the scanning space of size(128, 512, 512) containing nodules and the relevant labels,
	and use it to do dectection regression!

5. 3d-cnn.ipynb:
	Use the data generated from step 4 to train a 2d-detection  models.
# Tianchi-AI-Nodule
# Tianchi-AI-Nodule
