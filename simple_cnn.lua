require 'torch'
require 'dp'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

local utils = require('utils')
local build_model = require('build_model')


-- load the data
data_path = 'dataset/data.h5'
dset = utils.load_data(data_path)
dset = utils.preprocess_data(dset)

-- Print data size and shape
for k, v in pairs(dset) do
  if v:dim() > 1 then
  	print(k, '(', v:size(1), v:size(2), v:size(3), v:size(4), ')')
  else
	print(k, '(', v:size(1), ')')
  end
end


-- Sanity check 1: initial loss --> Passed!
-- generate some data for sanity check
require 'math'
local x = torch.randn(100, 1, 48, 48)
local y = torch.Tensor(100)
for i = 1, 100 do
  y[i] = math.random(1, 7)
end

-- Build a sample model
local convlayer_params = {['num_filters']= {32, 64}, ['filter_size']= {3, 3} ,['stride']={1, 1}, 
                          ['s_batch_norm']= true, ['pool_dims']= 2, ['pool_strides']= 2}
local affinelayer_params = {['hidden_dims']= {100}, ['batch_norm']= true,['dropout']= true}

local w_scale = 5e-2
model = build_model.full_conv_net(convlayer_params, affinelayer_params, w_scale)
--cudnn.convert(model, cudnn)
--model:cuda()
model:training()
print(model)

-- define log softmax criterion for loss computation
crit = nn.CrossEntropyCriterion() 


-- sanity check 1 result:
local sanity_scores = model:forward(x)
local sanity_data_loss = crit:forward(sanity_scores, y)
print('Initial loss =', sanity_data_loss , '(should be about log(7) = 1.945)')

-- Train realistic data
local num = 5000
small_dset = {}
small_dset.X_train = dset.X_train:narrow(1, 1, num)
small_dset.y_train = dset.y_train:narrow(1, 1, num)
small_dset.X_val = dset.X_val
small_dset.y_val = dset.y_val

local num_epoch = 20 
local batch_size = 150
local itr_per_epoch = math.max(math.floor(num / batch_size), 1)
local reg = 0
local num_iterations = itr_per_epoch * num_epoch
local config = {
  learningRate= 0.0001,
}


local params, gradParams = model:getParameters()
local t = 0

function f(w)

  gradParams:zero()    
  local X_batch, y_batch = utils.get_minibatch(small_dset.X_train, small_dset.y_train, batch_size)
    
  -- X_batch = X_batch:cuda() 
  --y_batch = y_batch:cuda() 
  assert(w == params)
  local scores = model:forward(X_batch)
  local data_loss = crit:forward(scores, y_batch)
  local dscores = crit:backward(scores, y_batch)
  model:backward(X_batch, dscores)
  
  -- add regularization to loss
  --data_loss = data_loss + reg/2.0 * torch.norm(params)^2

  -- add regularization to gradients
  --gradParams:add(reg, params)
  
  if t % itr_per_epoch == 0 then
    print(t,'/', num_iterations, data_loss, torch.abs(gradParams):mean())
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
        
    local train_acc = utils.check_accuracy(small_dset.X_train, small_dset.y_train, model, batch_size)
    local val_acc = utils.check_accuracy(small_dset.X_val, small_dset.y_val, model, batch_size)
    model:training()
    
    config.learningRate = config.learningRate * 0.95

    if val_acc > old_val_acc then
	--best_params, _ = params:copy()
        best_val_acc = val_acc
        old_val_acc = val_acc
    end

    print('train_acc: ', train_acc, 'val_acc: ', val_acc)
    print('\n')
        
  end

end

print('best val accuracy:', best_val_acc)
utils.save_model(model, 'first_model')
