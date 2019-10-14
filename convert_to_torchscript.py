import torch
from ddsp import NeuralSynth, IncrementalNS

NS = NeuralSynth()
INS = IncrementalNS(NS)

f0 = torch.randn(1,1,1)
lo = torch.randn(1,1,1)
hx = torch.randn(1,1,512)

traced = torch.jit.trace(INS, [f0, lo, hx])
traced.save("ddsp.torchscript")
