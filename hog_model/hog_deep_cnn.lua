require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

local utils = require('utils')
local build_model = require('build_hog_model')


-- load and preprocess the data
data_path = '../dataset/data.h5'
hog_path = '../dataset/hog_features.h5'
dset = utils.load_data(data_path)
dset = utils.preprocess_data(dset)
hog_feats = utils.load_hog(hog_path)


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
local hg = torch.randn(100, 325)
local y = torch.Tensor(100)
for i = 1, 100 do
  y[i] = math.random(1, 7)
end


----------------------------Build a model------------------------------------
-- 1. model parameters
local convlayer_params = {['num_filters']= {32, 64}, ['filter_size']= {3, 3} ,['stride']={1, 1}, 
                          ['s_batch_norm']= false, ['max_pooling'] = {true, true}, ['pool_dims']= 2, ['pool_strides']= 2, ['dropout']=false}
local affinelayer_params = {['hidden_dims']= {256}, ['batch_norm']= false, ['dropout']= false}

local w_scale = 1e-3

-- 2. build model
conv_model, D = build_model.conv_net(convlayer_params, w_scale)
--cudnn.convert(model, cudnn)
--model:cuda()
conv_model:training()
print(conv_model)

fan_in = D + hog_feats.X_train:size(2)
fc_model = build_model.fc_net(affinelayer_params, fan_in, w_scale)
fc_model:training()
print(fc_model)

-- define log softmax criterion for loss computation
crit = nn.CrossEntropyCriterion()
--crit:cuda()
------------------------------------------------------------------------------

-- sanity check 1 result:
local sanity_conv_output = conv_model:forward(x)
local sanity_fc_input = torch.cat(sanity_conv_output, hg, 2)

local sanity_scores = fc_model:forward(sanity_fc_input)
local sanity_loss = crit:forward(sanity_scores, y)
print('Initial loss:', sanity_loss, 'should be around log(7)') -- PASSED! :)
-----------------------------Train realistic data-------------------------------

local num = 100 --dset.X_train:size(1)
small_dset = {}
small_dset.X_train = dset.X_train:narrow(1, 1, num)
small_dset.y_train = dset.y_train:narrow(1, 1, num)
small_dset.X_val = dset.X_val
small_dset.y_val = dset.y_val

small_hog = {}
small_hog.X_train = hog_feats.X_train:narrow(1, 1, num)
small_hog.X_val = hog_feats.X_val

local num_epoch = 100 
local batch_size = 50
local itr_per_epoch = math.max(math.floor(num / batch_size), 1)
local reg = 0.0
local num_iterations = itr_per_epoch * num_epoch
local config = {
  learningRate= 0.0001,
}


local loss_history = torch.Tensor(num_iterations):fill(0)
local conv_params, conv_gradParams = conv_model:getParameters()
local fc_params, fc_gradParams = fc_model:getParameters()

local params = torch.cat(conv_params, fc_params)
local t = 0 

function f(w)

  conv_gradParams:zero()
  fc_gradParams:zero()  
  
  local X_batch, y_batch, hog_batch = utils.get_hog_minibatch(small_dset.X_train, small_dset.y_train, small_hog.X_train, batch_size)
  
  X_batch = X_batch--:cuda() 
  y_batch = y_batch--:cuda() 
  assert(w == params)

  local conv_output = conv_model:forward(X_batch)
  local fc_input = torch.cat(conv_output, hog_batch, 2)

  local scores = fc_model:forward(fc_input)
  local data_loss = crit:forward(scores, y_batch)
  local dscores = crit:backward(scores, y_batch)

  local dfc_input = fc_model:backward(fc_input, dscores)
  dfc = dfc_input:narrow(2, 1, conv_output:size(2)):contiguous()
  conv_model:backward(X_batch, dfc)
  
  local gradParams = torch.cat(conv_gradParams, fc_gradParams)

  -- add regularization to loss
  data_loss = data_loss + reg/2.0 * torch.norm(params)^2
 
  -- add regularization to gradients
  gradParams:add(reg, params)
 
  loss_history[t] = data_loss
   
  if t % itr_per_epoch == 0 then
    print(string.format('%d / %d', t, num_iterations), string.format('loss: %f, grad_mean: %f', data_loss, torch.abs(gradParams):mean()))
  end 
   
  return data_loss, gradParams
end

-------------------------Training process-----------------------------
local best_params = torch.Tensor(params:size())
local train_acc = torch.Tensor(num_epoch)
local val_acc = torch.Tensor(num_epoch)
local best_val_acc = 0.
local epoch_counter = 1
print('Training started...\n')

timer = torch.Timer() 
while t < num_iterations do

  t = t + 1

  params = optim.adam(f, params, config)
  --optim.sgd(f, params, config)

  fc_params:copy(params:narrow(1, conv_params:size(1), fc_params:size(1)))
  conv_params:copy(params:narrow(1, 1, conv_params:size(1)))
  
  
  -- Check training and validation accuracy once in a while
  if t % itr_per_epoch == 0 or t == num_iterations then
        
    train_acc[epoch_counter] = utils.hog_check_accuracy(small_dset.X_train, small_dset.y_train, small_hog.X_train, 
                                                        conv_model, fc_model, batch_size)
    val_acc[epoch_counter] = utils.hog_check_accuracy(small_dset.X_val, small_dset.y_val, small_hog.X_val, 
                                                        conv_model, fc_model, batch_size)
    conv_model:training()
    fc_model:training()
    
    config.learningRate = config.learningRate * 0.95

    if val_acc[epoch_counter] > best_val_acc then
	best_params:copy(params)
        best_val_acc = val_acc[epoch_counter]
    end

    print(string.format('train_acc: %f , val_acc: %f', train_acc[epoch_counter], val_acc[epoch_counter]))
    print('\n')
    epoch_counter = epoch_counter +1
        
  end

end 
timer:stop()
print('Traing Done. Elapsed time: ' .. timer:time().real .. ' seconds')

-------------------------Test accuracy--------------------------
params:copy(best_params)
test_acc = utils.hog_check_accuracy(dset.X_test, dset.y_test, hog_feats.X_test, conv_model, fc_model, batch_size)

print('best val accuracy:', best_val_acc)
print('test accuracy:', test_acc)

-------------------------plot results---------------------------
print('plot results...')
-- loss--
gnuplot.pngfigure('loss_history.png')
gnuplot.plot(torch.range(1, num_iterations), loss_history)
gnuplot.xlabel('Iteration')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

--accuracy--
gnuplot.pngfigure('Training_history.png')
gnuplot.plot({'Training', torch.range(1, num_epoch), train_acc,  '-'},
   {'Validation', torch.range(1, num_epoch), val_acc, '-'})
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Accuracy')
gnuplot.plotflush()

-------------------------Store model---------------------------
print('Saving the trained model and history...')
torch.save('hog_model.bin', model)
-- to load model use: model = torch.load('file_name')

local training_hist = {

        num_epoch = num_epoch,
	num_iterations = num_iterations,
	loss_history = loss_history,
	train_acc = train_acc,
	val_acc = val_acc,
	test_acc = test_acc,
	best_val_acc = best_val_acc
}
torch.save('hog_train_hist.bin', training_hist)

print('Done! Bye ;) :)')
  
  
