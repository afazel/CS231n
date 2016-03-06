require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

local layer_num = 5

model = torch.load('model.bin')
cudnn.convert(model, nn)
print(model)

-- visualize the feature maps of the first layer 
features = model:get(layer_num).output

-- extract the first 10 images
--tmp= b[{{1,10}}]
for i= 1, 10 do
	file_name = string.format('layer%d_feature%d.png', layer_num, i)
	image.save(file_name, image.toDisplayTensor{input=features[i]})
end
