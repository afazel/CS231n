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
    --y_batch = torch.LongTensor():resize(y_batch:size()):copy(y_batch)
    num_correct = num_correct + torch.eq(y_pred, y_batch):sum()
    num_tested = num_tested + batch_size
  end
  return num_correct / num_tested

end


-- save model
local function save_model(model, out_file)

  local next_weight_idx = 1
  local next_bn_idx = 1
  local f = hdf5.open(out_file, 'w')
  for i = 1, #model do
    local layer = model:get(i)
    if torch.isTypeOf(layer, nn.SpatialConvolution) or 
       torch.isTypeOf(layer, nn.Linear) then
      f:write(string.format('/W%d', next_weight_idx), layer.weight:float())
      f:write(string.format('/b%d', next_weight_idx), layer.bias:float())
      next_weight_idx = next_weight_idx + 1
    elseif torch.isTypeOf(layer, nn.SpatialBatchNormalization) or
           torch.isTypeOf(layer, nn.BatchNormalization) then
      f:write(string.format('/gamma%d', next_bn_idx), layer.weight:float())
      f:write(string.format('/beta%d', next_bn_idx), layer.bias:float())
      f:write(string.format('/running_mean%d', next_bn_idx), layer.running_mean:float())
      if torch.isTypeOf(layer, nn.BatchNormalization) then
        f:write(string.format('/running_var%d', next_bn_idx),
                torch.pow(layer.running_std, -2.0):add(-layer.eps):float())
      elseif torch.isTypeOf(layer, nn.SpatialBatchNormalization) then
        f:write(string.format('/running_var%d', next_bn_idx),
                layer.running_var:float())
      end
      next_bn_idx = next_bn_idx + 1
    end
  end
  f:close()
end



M.load_data = load_data
M.preprocess_data = preprocess_data
M.get_minibatch = get_minibatch
M.check_accuracy = check_accuracy
M.save_model = save_model

return M
