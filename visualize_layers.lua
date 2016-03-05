require 'nn'
require 'torch'

model = torch.load('first_model.bin')

-- visualize the feature maps of the first layer 
features = model:get(1).output

-- extract the first 10 images
--tmp= b[{{1,10}}]
image.save("feature.png", image.toDisplayTensor{input=features[1]})
