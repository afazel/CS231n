require 'torch'
require 'dp'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

path = 'dataset/data.h5'

local function load_data(data_file)
  local f = hdf5.open(data_file)
  local dset = {}
  dset.X_train = f:read('X_train'):all()
  dset.y_train = f:read('y_train'):all() +1
  dset.X_val = f:read('X_val'):all()
  dset.y_val = f:read('y_val'):all() + 1
  dset.X_test = f:read('X_test'):all()
  dset.y_test = f:read('y_test'):all() + 1
  f:close()
  print('Data is loaded!')
    
  return dset
end

--[[ preprocess data before feeding that into the convnet
     subtract mean of training data, and reshape it to the
     form (N, 1, H, W) ]]--
local function preprocess_data(dset)

  local dset_new = {}
    
    -- subtract mean
  mean_image = torch.mean(dset.X_train)
  dset_new.X_train = dset.X_train - mean_image
  dset_new.X_val = dset.X_val - mean_image
  dset_new.X_test = dset.X_test - mean_image

  -- reshape
  H = dset.X_train:size(2)
  W = dset.X_train:size(3)
  
  dset_new.X_train = torch.reshape(dset_new.X_train, dset.X_train:size(1), 1, H, W)
  dset_new.X_val = torch.reshape(dset_new.X_val, dset.X_val:size(1), 1, H, W)
  dset_new.X_test = torch.reshape(dset_new.X_test, dset.X_test:size(1), 1, H, W)
    
  dset_new.y_train = dset.y_train
  dset_new.y_val = dset.y_val
  dset_new.y_test = dset.y_test

  print('Data preprocessing done. ready to go!')
  return dset_new
end

local function get_minibatch(X, y, batch_size)

  local mask = torch.LongTensor(batch_size):random(X:size(1))
  local X_batch = X:index(1, mask)
  local y_batch = y:index(1, mask)
  return X_batch, y_batch

end

local function check_accuracy(X, y, model, batch_size)
   
  --[[ change the model mode to evaluation, Improtant for dropout 
  and batchnormalization ]]--
  model:evaluate()
  local num_correct = 0
  local num_tested = 0
    
  for t = 1, 20 do
    local X_batch, y_batch = get_minibatch(X, y, batch_size)
    
    --X_batch = X_batch:cuda()
    --y_batch = y_batch:cuda()
    local scores = model:forward(X_batch)
    local _, y_pred = scores:max(2)
    num_correct = num_correct + torch.eq(y_pred, y_batch):sum()
    num_tested = num_tested + batch_size
  end
  return num_correct / num_tested

end

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

-----------------------------Execution------------------------------------
-- load the data
local dset = load_data(path)
dset = preprocess_data(dset)

-- Print data size and shape
for k, v in pairs(dset) do
  if v:dim() > 1 then
  	print(k, '(', v:size(1), v:size(2), v:size(3), v:size(4), ')')
  else
	print(k, '(', v:size(1), ')')
  end
end

-- Build a sample model
--[[local convlayer_params = {['num_filters']= {32, 32}, ['filter_size']= {3, 3} ,['stride']={1, 1}, 
                          ['s_batch_norm']= false, ['pool_dims']= 2, ['pool_strides']= 2}
local affinelayer_params = {['hidden_dims']= {50}, ['batch_norm']= false,['dropout']= false}

model = full_conv_net(convlayer_params, affinelayer_params)
model:training()
print('Model is generated. The architecture is:')
print('\n')
print(model)
print('\n')]]--

-- compute loss and gradients
-- define log softmax criterion for loss computation
crit = nn.CrossEntropyCriterion() 

-- Sanity check 1: initial loss
local convlayer_params = {['num_filters']= {32, 32}, ['filter_size']= {3, 3} ,['stride']={1, 1}, 
                          ['s_batch_norm']= true, ['pool_dims']= 2, ['pool_strides']= 2}
local affinelayer_params = {['hidden_dims']= {100}, ['batch_norm']= true,['dropout']= false}

local w_scale = 5e-2
model = full_conv_net(convlayer_params, affinelayer_params, w_scale)
model:training()

-- generate some data
require 'math'
local x = torch.randn(100, 1, 48, 48)
local y = torch.Tensor(100)
for i = 1, 100 do
	y[i] = math.random(1, 7)
end

local sanity_scores = model:forward(x)
local sanity_data_loss = crit:forward(sanity_scores, y)
print('Initial loss =', sanity_data_loss)

-- Train data
local num = 5000
small_dset = {}
small_dset.X_train = dset.X_train:narrow(1, 1, num)
small_dset.y_train = dset.y_train:narrow(1, 1, num)
small_dset.X_val = dset.X_val
small_dset.y_val = dset.y_val

local num_epoch = 30 
local batch_size = 150
local itr_per_epoch = math.max(math.floor(num / batch_size), 1)
local reg = 0
local num_iterations = itr_per_epoch * num_epoch
local config = {
  learningRate= 0.0001,
}

print(model)
local params, gradParams = model:getParameters()
local t = 0

function f(w)

  gradParams:zero()    
  local X_batch, y_batch = get_minibatch(small_dset.X_train, small_dset.y_train, batch_size)
    
  -- X_batch = X_batch:cuda() 
  --y_batch = y_batch:cuda() 
  assert(w == params)
  local scores = model:forward(X_batch)
  local data_loss = crit:forward(scores, y_batch)
  local dscores = crit:backward(scores, y_batch)
  
  --------------- manually compute dscores ------------------
  local max_scores, _ = torch.max(scores, 2)
  local N = scores:size(1)
  local probs = torch.Tensor(N, 7):zero()
  for i = 1, N do
  	for j = 1, 7 do 
  		probs[i][j] = torch.exp(scores[i][j] - max_scores[i][1])
 	end
        probs[i] = torch.div(probs[i], torch.sum(probs[i]))

  end

  local dx = probs:clone()
  for i = 1, N do
	dx[i][ y_batch[i]] = dx[i][ y_batch[i]] - 1
  end

  dx = dx / N
  --print(torch.sqrt(torch.sum((dx - dscores):cmul(dx - dscores))/(dx:size(1)*dx:size(2))))
  --------------------------------------------------------------

  model:backward(X_batch, dscores)
  
  -- add regularization to loss
  --data_loss = data_loss + reg/2.0 * torch.norm(params)^2

  -- add regularization to gradients
  --gradParams:add(reg, params)
  
  if t % itr_per_epoch == 0 then
    print(t, '/', num_iterations, data_loss, torch.abs(gradParams):mean())
  end
  
  return data_loss, gradParams
end

-- optimization process
local old_val_acc = 0.
local best_val_acc = 0.
print('Training started...\n')
while t < num_iterations do
    
  t = t + 1
  optim.adam(f, params, config)
  --optim.sgd(f, params, config)

  -- Check training and validation accuracy once in a while
  if t % itr_per_epoch == 0 then
        
    local train_acc = check_accuracy(small_dset.X_train, small_dset.y_train, model, batch_size)
    local val_acc = check_accuracy(small_dset.X_val, small_dset.y_val, model, batch_size)
    model:training()
    
    config.learningRate = config.learningRate * 0.95

    if val_acc > old_val_acc then
	--best_params, _ = params:copy()
        best_val_acc = val_acc
        old_val_acc = val_acc
    end

    print('train acc: ', train_acc, 'val_acc: ', val_acc)
    print('\n')
        
  end

end

print('best val accuracy:', best_val_acc)
