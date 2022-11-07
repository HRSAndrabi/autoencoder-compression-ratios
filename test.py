l1 = [2, 4, 5, 6]
l2 = [1, 2, 2, 3]

l3 = [instance[0] for instance in zip(l1, l2) if instance[1] == 2]
print(l3)
print(l2.index(2))