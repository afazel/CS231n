
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'
require 'math'

local utils = require('utils')
local build_model = require('build_model')


-- load and preprocess the data
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

----------------------------Build a model------------------------------------
-- 1. model parameters
local convlayer_params = {['num_filters']= {64, 128, 512, 512}, ['filter_size']= {3, 5, 3, 3} ,['stride']={1, 1, 1, 1},
                          ['s_batch_norm']= true, ['max_pooling'] = {true, true, true, true}, ['pool_dims']= 2, ['pool_strides']= 2, ['dropout']={true}}
local affinelayer_params = {['hidden_dims']= {256, 512}, ['batch_norm']= true, ['dropout']= true}

local w_scale = 1e-3

-- 2. build model
model = build_model.full_conv_net(convlayer_params, affinelayer_params, w_scale)
cudnn.convert(model, cudnn)
model:cuda()
model:training()
print(model)

-- define log softmax criterion for loss computation
crit = nn.CrossEntropyCriterion()
crit:cuda()

-----------------------------Train realistic data-------------------------------
local num = 2000 --dset.X_train:size(1)
small_dset = {}
small_dset.X_train = dset.X_train:narrow(1, 1, num)
small_dset.y_train = dset.y_train:narrow(1, 1, num)
small_dset.X_val = dset.X_val
small_dset.y_val = dset.y_val

local num_epoch = 1 
local batch_size = 50
local itr_per_epoch = math.max(math.floor(num / batch_size), 1)
local reg = {1e-7, 1e-6}
local lr = {1e-3, 1e-2}
local num_iterations = 500 --itr_per_epoch * num_epoch

local params, gradParams = model:getParameters()


local best_params = torch.Tensor(params:size())
local best_val_acc = 0.
local best_train_acc = 0.
local best_ever_acc = 0.
local best_acc_hist = torch.Tensor(#lr, #reg)
local train_acc_hist = torch.Tensor(#lr, #reg)

for i = 1, #lr do
  for j = 1, #reg do

        model:reset()
        print(string.format('model for lr=%f, reg=%f', lr[i], reg[j]))
        best_val_acc = 0.
	local config = {
  		learningRate= lr[i],
	}


	local t = 0 

	function f(w)

  		gradParams:zero()    
  		local X_batch, y_batch = utils.get_minibatch(small_dset.X_train, small_dset.y_train, batch_size)
    
  		X_batch = X_batch:cuda() 
  		y_batch = y_batch:cuda() 
  		assert(w == params)
  		local scores = model:forward(X_batch)
  		local data_loss = crit:forward(scores, y_batch)
  		local dscores = crit:backward(scores, y_batch)
  		model:backward(X_batch, dscores)
  
  		-- add regularization to loss
  		data_loss = data_loss + reg[j]/2.0 * torch.norm(params)^2

  		-- add regularization to gradients
  		gradParams:add(reg[j], params)
  
  		if t % itr_per_epoch == 0 then
    			print(string.format('%d / %d', t, num_iterations), string.format('loss:%f , mean(grads):%f' ,data_loss, torch.abs(gradParams):mean()))
  		end
  
  		return data_loss, gradParams
	end
        
        --start training--
	print('Training started...\n')

	while t < num_iterations do
    
  		t = t + 1
  		optim.adam(f, params, config)
  		--optim.sgd(f, params, config)

  		-- Check training and validation accuracy once in a while
  		if t % itr_per_epoch == 0 or t == num_iterations then
        
    			local train_acc = utils.check_accuracy(small_dset.X_train, small_dset.y_train, model, batch_size)
    			local val_acc = utils.check_accuracy(small_dset.X_val, small_dset.y_val, model, batch_size)
    			model:training()
    
    			config.learningRate = config.learningRate * 0.95

    			if val_acc > best_val_acc then
        			best_val_acc = val_acc
                                best_train_acc = train_acc
			end	
			if val_acc > best_ever_acc then
				best_ever_acc = best_val_acc
                                print('params check BEFORE:', torch.abs(best_params):mean())	
				best_params:copy(params)
                                print('params check AFTER:', torch.abs(params):mean(), torch.abs(best_params):mean())
			end

    			print(string.format('reg=%f, lr=%f', lr[i], reg[j]), string.format('train acc:%f , val_acc:%f', train_acc, val_acc))
    			print('\n')
        
  		end

	end
	
        best_acc_hist[i][j] = best_val_acc
        train_acc_hist[i][j] = best_train_acc
         
   end
end
-------------------------Test accuracy--------------------------
print('params check BEFORE:', torch.abs(params):mean())
params:copy(best_params)
print('params check AFTER:', torch.abs(params):mean())
test_acc = utils.check_accuracy(dset.X_test, dset.y_test, model, batch_size)

print('best val accuracy:', best_ever_acc)
print('best test accuracy:', test_acc)

torch.save('best_model.bin', model)

local cv_hist = {
	reg = reg,
	lr = lr,
	best_acc_hist = best_acc_hist,
        train_acc_hist = train_acc_hist
}
torch.save('models_best_acc.bin', cv_hist)

print('Done! Bye ;) :)')
  
  
