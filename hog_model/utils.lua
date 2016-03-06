require 'torch'
require 'dp'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'

local M = {}

-- A function to load the data from h5 file:

local function load_data(data_file_path)
  local f = hdf5.open(data_file_path)
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

local function load_hog(hog_path)
  local f = hdf5.open(hog_path)

  local hog_features = {}
  hog_features.X_train = f:read('X_train_feats'):all()
  hog_features.X_val = f:read('X_val_feats'):all()
  hog_features.X_test = f:read('X_test_feats'):all()

  return hog_features 
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

local function get_hog_minibatch(X, y, hog, batch_size)

  local mask = torch.LongTensor(batch_size):random(X:size(1))
  local X_batch = X:index(1, mask)
  local hog_batch = hog:index(1, mask)
  local y_batch = y:index(1, mask)
  return X_batch, y_batch, hog_batch

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
    --y_batch = torch.LongTensor():resize(y_batch:size()):copy(y_batch)
    num_correct = num_correct + torch.eq(y_pred, y_batch):sum()
    num_tested = num_tested + batch_size
  end
  return num_correct / num_tested

end

local function hog_check_accuracy(X, y, hog, model1, model2, batch_size)
   
  --[[ change the model mode to evaluation, Improtant for dropout 
  and batchnormalization ]]--
  model1:evaluate()
  model2:evaluate()
  local num_correct = 0
  local num_tested = 0
    
  for t = 1, 20 do
    local X_batch, y_batch, hog_batch = get_hog_minibatch(X, y, hog, batch_size)
    
    --X_batch = X_batch:cuda()
    --y_batch = y_batch:cuda()
    local conv_output = model1:forward(X_batch)
    local fc_input = torch.cat(conv_output, hog_batch, 2)
    local scores = model2:forward(fc_input)
    local _, y_pred = scores:max(2)
    --y_batch = torch.LongTensor():resize(y_batch:size()):copy(y_batch)
    num_correct = num_correct + torch.eq(y_pred, y_batch):sum()
    num_tested = num_tested + batch_size
  end
  return num_correct / num_tested

end


M.load_data = load_data
M.load_hog = load_hog
M.preprocess_data = preprocess_data
M.get_minibatch = get_minibatch
M.get_hog_minibatch = get_hog_minibatch
M.check_accuracy = check_accuracy
M.hog_check_accuracy = hog_check_accuracy

return M
