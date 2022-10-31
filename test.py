from enum import Enum
import scipy.special


class Sigmoid:
	def activation(self, x):
		return scipy.special.expit(x)

	def derrivative(self, x):
		return x * (1 - x)


class ActivationFunctions(Enum):
	sigmoid = Sigmoid()

activation_func = ActivationFunctions["sigmoid"]


print(ActivationFunctions._member_names_)

print(list(set(["sigmoid", "sigmoid", "relu"]).difference(ActivationFunctions._member_names_)))