require 'torch'
require 'dp'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

local N = {}

--build a convnet model
local function full_conv_net(convlayer_params, affinelayer_params, w_scale)
  
  local num_filters = convlayer_params['num_filters'] --{64,  64,  128,  128,  256, 256, 512, 512, 1024}
  local filter_sizes =  convlayer_params['filter_size'] --{5,   3,   3,   3,   3,   3,   3,   3,   3}
  local filter_strides = convlayer_params['stride'] --{2,   1,   2,   1,   2,   1,   2,   1,   2}
  local use_sbatchnorm = convlayer_params['s_batch_norm']
  local maxpool_dim = convlayer_params['pool_dims']
  local maxpool_stride = convlayer_params['pool_strides']
  
  local hidden_dims = affinelayer_params['hidden_dims']
  local use_batchnorm = affinelayer_params['batch_norm']
  local use_dropout = affinelayer_params['dropout']
    
  local num_classes = 7
  -- C: number of channels , H,W: height and width of an image
  
  local C, H, W = 1, 48, 48 
  
  local next_C = C
  local next_H = H
  local next_W = W

  -- add layers
  local layer_counter = 0
  local model = nn.Sequential()
  local m = model.modules
  for i = 1, #num_filters do
    local zero_pad = (filter_sizes[i] - 1) / 2

    model:add(nn.SpatialConvolution(next_C, num_filters[i], filter_sizes[i], filter_sizes[i], 
                filter_strides[i], filter_strides[i], zero_pad, zero_pad))

    -- Manually initialize bias and weights
    layer_counter = layer_counter + 1 
    m[layer_counter].bias:fill(0)
    m[layer_counter].weight:randn(num_filters[i], next_C, filter_sizes[i], filter_sizes[i])
    m[layer_counter].weight:div(w_scale)

    -- data size after conv layer operation   
    next_C = num_filters[i]
    next_W = (next_W + 2*zero_pad - filter_sizes[i]) / filter_strides[i] + 1
    next_H = (next_H + 2*zero_pad - filter_sizes[i]) / filter_strides[i] + 1
        
    if use_sbatchnorm then
        model:add(nn.SpatialBatchNormalization(next_C))
        layer_counter = layer_counter + 1
        m[layer_counter].weight:fill(1.0)
        m[layer_counter].bias:fill(0.0)
    end
            
    model:add(nn.ReLU())
    layer_counter = layer_counter + 1

    model:add(nn.SpatialMaxPooling(maxpool_dim, maxpool_dim, maxpool_stride, maxpool_stride))
    layer_counter = layer_counter + 1
    
    -- data size after max pooling operation
    next_W = (next_W - maxpool_dim) / maxpool_stride + 1
    next_H = (next_H - maxpool_dim) / maxpool_stride + 1
  end
    
  local next_D = next_C * next_W * next_H
  model:add(nn.View(-1):setNumInputDims(3))
  layer_counter = layer_counter + 1
      
  for i = 1, #hidden_dims do

    model:add(nn.Linear(next_D, hidden_dims[i]))
    layer_counter = layer_counter + 1
    m[layer_counter].bias:fill(0)
    --m[layer_counter].weight:randn(next_D, hidden_dims[i]) 
    --m[layer_counter].weight:div(w_scale)
    next_D = hidden_dims[i]
    
    if use_batchnorm then
      model:add(nn.BatchNormalization(hidden_dims[i]))
      layer_counter = layer_counter + 1
      m[layer_counter].weight:fill(1.0)
      m[layer_counter].bias:fill(0.0)
    end
    
    if use_dropout then
      model:add(nn.Dropout(0.5))
      layer_counter = layer_counter + 1
    end
                    
    model:add(nn.ReLU())
    layer_counter = layer_counter + 1

  end
                
  model:add(nn.Linear(next_D, num_classes))
  layer_counter = layer_counter + 1
  m[layer_counter].bias:fill(0)
  --m[layer_counter].weight:randn(next_D, num_classes)
  --m[layer_counter].weight:div(w_scale)

  return model
end

N.full_conv_net = full_conv_net

return N

