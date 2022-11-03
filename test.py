import matplotlib.pyplot as plt
from src.autoencoder import AutoEncoder 

colors = ["#FF412C", "#003CAB", "#9F29FF", "#C20800", "#31CAA8", "#06A6EE", "#000000"]
markers = ["s", "o", "^", "P", "o", "X", "^", "v"]

with plt.style.context("science"):
	plt.figure(figsize=(9,3), dpi=300)
	for hl_i, hl in enumerate([2, 4, 8, 16, 32, 64, 128]):
		for bias_i, bias in enumerate([True, False]):
			if bias:
				model_name = f"hl_{hl}_bias"
			else:
				model_name = f"hl_{hl}"
			model = AutoEncoder()
			model.load(
				path_to_spec=f"./src/models/{model_name}.json",
				print_summary=False,
			)
			mse = model.spec["loss"]
			plt.plot(
				[*range(0, len(mse))], 
				mse,
				color=colors[hl_i],
				marker=markers[bias_i],
				markerfacecolor="#ffffff",
				markersize=3,
			)

	plt.legend()
	plt.show()