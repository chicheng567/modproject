import torch
hid1 = torch.load("/modproject/hidden_states_0.pt")
hid2 = torch.load("/modproject/hidden_states_.pt")
print(hid1 == hid2)
