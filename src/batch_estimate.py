import pandas as pd
from autoencoder import AutoEncoder

def load_mnist_data(file_path:str, nrows:int=0) -> tuple[list, list]:
	"""Reads in MNIST dataset, aquired from: 
	https://www.kaggle.com/datasets/oddrationale/mnist-in-csv.

	Args:
		file_path (str): file path to the dataset.
		nrows (int, optional): number of rows to read in. 
		Defaults to all rows.

	Returns:
		tuple[list, list]: image instances, and corresponding 
		one-hot encoded labels.
	"""
	if nrows:
		df = pd.read_csv(filepath_or_buffer=file_path, nrows=nrows)
	else: 
		df = pd.read_csv(filepath_or_buffer=file_path)
	x = ((df[df.columns[1:]]/255 * 0.99) + 0.01).to_numpy()
	y = pd.get_dummies(df["label"]).to_numpy()
	return x, y

x_train, y_train = load_mnist_data(
	file_path="src/data/mnist_train.csv",
)
x_test, y_test = load_mnist_data(
	file_path="src/data/mnist_test.csv",
	nrows=100,
)

base_spec = {
	"name": None,
	"layers": [
		{ "type": "input", "nodes": 784 },
		{ "type": "hidden", "nodes": None, "activation_func": "sigmoid" },
		{ "type": "output", "nodes": 784, "activation_func": "sigmoid" },
	],
	"bias": False,
}

for hl in [2, 4, 8, 16, 32, 64, 128]:
	spec = base_spec
	spec["name"] = f"hl_{hl}_bias"
	spec["layers"][1]["nodes"] = hl
	spec["bias"] = True
	model = AutoEncoder(spec, 0.05)
	model.train(instances=x_train, epochs=50)
	model.save(dir_path="./src/models/")