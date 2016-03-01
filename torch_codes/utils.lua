require 'torch'
require 'dp'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

local cmd = torch.CmdLine()
cmd:option('-input', path)
cmd:option('-output_h5', 'torch_model.h5')
cmd:option('-output_t7', 'torch_model.t7')
local opt = cmd:parse(arg or {})


local function load_data(data_file)
  print(data_file)
  local f = hdf5.open(data_file)
  local dset = {}
  dset.X_train = f:read('X_train'):all()
  dset.y_train = f:read('y_train'):all() + 1
  dset.X_val = f:read('X_val'):all()
  dset.y_val = f:read('y_val'):all() + 1
  dset.X_test = f:read('X_train'):all()
  dset.y_test = f:read('y_train'):all() + 1
  f:close()
  print('Data is loaded, Ready to go!')
  return dset
end

--[[ preprocess data before feeding that into the convnet
     subtract mean of training data, and reshape it to the
     form (N, 1, H, W) ]]--
local function preprocess_data(dset)

    dset_new = {}

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

    print('data preprocessing done.')
	return dset_new
end

--[[build a model]]--


--[[ Now Let's build and train the model 
local convlayer_params = {'convlayer_num'= 2 ,'num_filters'= {32, 32}, 'filter_size'= 3,
                   'stride'=1, 'pad'=(3 - 1) / 2, 's_batch_norm'= false,
                   'pool_dim'= 2, 'pool_stride'= 2}
local affinelayer_params = {'afflayer_num'= 1, 'hidden_dim'= [50], 'batch_norm'= false,
                      'dropout'= false}]]--

--local dset = load_data(path)
--dset = preprocess_data(dset)