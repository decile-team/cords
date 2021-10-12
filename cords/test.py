import torch

t = torch.rand(50, 100)
chunked_t = torch.chunk(t, 5, dim=0)
new_t = []
for i in range(len(chunked_t)):
    new_t.append(torch.mean(chunked_t[i], dim=0).view(1, -1))
new_t = torch.cat(new_t, dim=0)
print()