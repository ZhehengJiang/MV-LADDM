import argparse
import tensorflow as tf
keras=tf.contrib.keras
from src.MV_LADDM_model import MV_LADDM

import os

if __name__ == "__main__":

	# Setup argument parser
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	parser = argparse.ArgumentParser()
	parser.add_argument("-train", default=False, action="store_true" , help="Flag to intiate training")
	parser.add_argument("-test", default=False, action="store_true" , help="Flag to initiate testing")
	parser.required=True
	args = parser.parse_args()

	model = MV_LADDM(classes=13, MLVAE_latent_dim=1000, num_cameras=2, sequence_length=30)
	model.load_data()

	if args.test:
		model.test_model()
	else:
		model.train_model()
