from thop import profile
import torch
model_list = []

import sanet_S as Model


model_list.append(Model.get_pred_model())

input = torch.randn(1, 3, 1024, 2048)
for model in model_list:
    print(model)
    flops, params = profile(model, inputs=(input,))
    print("FLOPs: %.2fG" % (flops / 1e9))
    print("Number of parameter: %.2fM" % (params / 1e6))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))