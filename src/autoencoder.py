import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import json
import copy
from enum import Enum

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Sigmoid:
	"""Implementation of the sigmoid function.
	"""
	def activation(self, x):
		return scipy.special.expit(x)

	def derrivative(self, x):
		return x * (1 - x)


class ActivationFunctions(Enum):
	"""Enumeration to keep track of implemented activation 
	functions for initial model validation.
	"""
	sigmoid = Sigmoid()


class AutoEncoder:
	"""Abstract implementation of an autoencoder network.
	"""

	def __init__(self, spec:dict=None, learning_rate:int=0.05):
		if spec:
			if not "input" in [layer["type"] for layer in spec["layers"]]:
				raise ValueError("Models must contain at least one input layer.")
			if not "hidden" in [layer["type"] for layer in spec["layers"]]:
				raise ValueError("Models must contain at least one hidden layer.")
			if not "output" in [layer["type"] for layer in spec["layers"]]:
				raise ValueError("Models must contain at least one output layer.")
			if 0 in [layer["nodes"] for layer in spec["layers"]]:
				raise ValueError("All layers must contain at least one node.")
			if len(list(set([layer["activation_func"] for layer in spec["layers"][1:]]).difference(ActivationFunctions._member_names_))):
				raise NotImplementedError(
					f"One or more specified activation functions are not implemented. Implemented functions include: {', '.join(ActivationFunctions._member_names_)}"
				)

			self.spec = spec
			self.spec["loss"] = []
			self.learning_rate = learning_rate
			
			for i, layer in enumerate(spec["layers"]):
				if layer["type"] != "output":
					next_layer = spec["layers"][i+1]
					self.spec["layers"][i]["weights"] = np.random.normal(
						loc=0.0, 
						scale=pow(next_layer["nodes"], -0.5), 
						size=(next_layer["nodes"], layer["nodes"])
					)
					if self.spec.get("bias", False):
						self.spec["layers"][i]["bias"] = np.random.normal(
							loc=0.0, 
							scale=pow(next_layer["nodes"], -0.5), 
							size=(next_layer["nodes"],1)
						)
			self.model_summary()
	
	def train(self, train_instances:list, test_instances:list, epochs:int=1) -> None:
		"""Trains model using input instances and labels.

		Args:
			train_instances (list): tensor of input instances.
			test_instances (list): tensor of test instances.
			epochs (int, optional): number of epochs for which to
			train the model. Defaults to 1.
		"""
		print("\n".join([
			"========================================",
			"Training Model",
			"========================================",
		]))
		for epoch in range(epochs):
			print(f"Epoch: {epoch}/{epochs}")
			for i, instance in enumerate(train_instances):
				if (((i+1)/len(train_instances)) * 100) % 25 == 0:
					print(f"Progress: {((i+1)/len(train_instances)) * 100}%")
				
				activations = self.forward_pass(instance)
				self.backpropogate(instance, activations)
			
			self.spec["loss"].append(
				self.evaluate(test_instances)
			)

	def forward_pass(self, instance:list) -> list:
		"""Conducts a forward pass through the model,
		generating activations for hidden and output
		layer(s).

		Args:
			instance (list): instance for which to conduct forward 
			pass.

		Returns:
			list: list of dictionaries containing activations
			at each layer.
		"""
		activations = copy.deepcopy(self.spec["layers"])
		for i, layer in enumerate(activations):
			if layer["type"] == "input":
				activations[i]["activations"] = np.array(instance, ndmin=2)
			else:
				activation_func = ActivationFunctions[layer["activation_func"]].value
				bias = activations[i-1].get("bias", 0)
				inputs = np.dot(activations[i-1]["weights"], activations[i-1]["activations"].T) - bias
				activations[i]["activations"] = activation_func.activation(inputs).T
		return activations

	def backpropogate(self, instance:list, activations:list) -> None:
		"""Adjusts model weights through backpropogation.
		Implemented backpropogation assumes sigmoid activation
		functions.

		Args:
			instance (list): instance for which to backpropogate.
			activations (list): list of dictionaries containing activations
			at each layer. Output by self.forward_pass.
		"""
		for i, layer in reversed(list(enumerate(activations))[1:]):
			if layer["type"] == "output":
				activations[i]["errors"] = np.array(instance, ndmin=2).T - layer["activations"].T
			else:
				activations[i]["errors"] = np.dot(layer["weights"].T, activations[i+1]["errors"])

			activation_func = ActivationFunctions[layer["activation_func"]].value
			
			self.spec["layers"][i-1]["weights"] += self.learning_rate * (
				np.dot(
					activations[i]["errors"] * activation_func.derrivative(activations[i]["activations"].T),
					activations[i-1]["activations"]
				)
			)
			if self.spec["bias"]:
				self.spec["layers"][i-1]["bias"] += self.learning_rate * (
					activations[i]["errors"] * activation_func.derrivative(activations[i]["activations"].T) * -1
				)

	def query(self, instance:list) -> None:
		"""Queries the trained model to encode and reconstruct
		an input instance.

		Args:
			instance (list): instance to resconstruct.
		"""
		activations = self.forward_pass(instance)
		reconstructed_instance = activations[-1]["activations"].T

		fig, axes = plt.subplots(figsize=(10, 10), nrows=1, ncols=2, sharey=True, sharex=True)
		for ax, img in zip(axes.flatten(), [instance, reconstructed_instance]):
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			im = ax.imshow(np.array(img, ndmin=2).T.reshape(28,28), cmap = plt.get_cmap("gray"))  
	
		
	def evaluate(self, instances:list) -> float:
		"""Encodes and decodes a set of instances and calculates reconstruction
		errors as mean-squared error of pixel value deviations.

		Args:
			instance (list): instance for which to backpropogate.

		Returns:
			float: average mean-squared error.
		"""
		errors = []
		for instance in instances:
			activations = self.forward_pass(instance)
			prediction = activations[-1]["activations"]
			errors.append((np.square(instance - prediction)).mean(axis=0))

		avg_mse = np.asarray(errors).mean()
		print("Average MSE: {}".format(avg_mse))
		return avg_mse

	def save(self, dir_path:str) -> None:
		"""Writes trained model spec to json.
		
		Args:
			dir_path (str): path to directory in which to save model.
		"""
		with open(f"{dir_path}{self.spec['name']}.json", "w") as f:
			json.dump(self.spec, f, cls=NumpyEncoder)
			print(f"Model saved to {dir_path}{self.spec['name']}.json.")

	def load(self, path_to_spec:str, print_summary:bool=True) -> dict:
		"""Loads in a trained model from json specification file.

		Args:
			path_to_spec (str): path to json specification file.
			print_summary (bool, optional): whether or not to print
			summary of model when it is loaded. Defaults to True.

		Returns:
			dict: trained model specification.
		"""
		with open(path_to_spec) as f:
			spec = json.load(f)
			for i, layer in enumerate(spec["layers"][:-1]):
				spec["layers"][i]["weights"] = np.asarray(spec["layers"][i]["weights"])
			self.spec = spec
			if print_summary:
				self.model_summary()

	def model_summary(self):
		print("\n".join([
			"========================================",
			"Model summary: {}".format(self.spec["name"]),
			"========================================",
			*["{} layer:	{} nodes".format(layer["type"].title(), layer["nodes"]) for layer in self.spec["layers"]],
			"========================================",
		]))