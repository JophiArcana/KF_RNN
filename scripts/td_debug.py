import torch
from tensordict import TensorDict

# Create two sample TensorDicts
td1 = TensorDict({
    "a": torch.tensor([1, 2, 3]),
    "b": torch.tensor([4, 5, 6]),
    "c": torch.tensor([7, 8, 9]),
}, batch_size=[3])

td2 = TensorDict({
    "b": torch.tensor([10, 11, 12]),
    "c": torch.tensor([13, 14, 15]),
    "d": torch.tensor([16, 17, 18]),
}, batch_size=[3])

# Get the sets of keys
keys1 = set(td1.keys())
keys2 = set(td2.keys())

# Find keys that are only in td1 (exclusive to td1)
exclusive_to_td1 = keys1 - keys2
print(f"Exclusive to td1: {exclusive_to_td1}")

# Find keys that are only in td2 (exclusive to td2)
exclusive_to_td2 = keys2 - keys1
print(f"Exclusive to td2: {exclusive_to_td2}")

# Find all keys that are exclusive (in either td1 or td2, but not both)
all_exclusive_keys = keys1.symmetric_difference(keys2)
print(f"All exclusive keys: {all_exclusive_keys}")

# Find common (intersecting) keys
common_keys = keys1.intersection(keys2)
print(f"Common keys: {common_keys}")

# Create a new TensorDict with only the keys exclusive to td1
td_exclusive_to_td1 = td1.select((*exclusive_to_td1,))
print(f"td_exclusive_to_td1:\n{td_exclusive_to_td1}")

# Create a new TensorDict with only the keys exclusive to td2
td_exclusive_to_td2 = td2.select((*exclusive_to_td2,))
print(f"td_exclusive_to_td2:\n{td_exclusive_to_td2}")




