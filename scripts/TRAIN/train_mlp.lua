package.path = arg[0]:dirname() .. "/../?.lua;" .. package.path
local common     = require "common"
local preprocess = require "PREPROCESS.load_train_and_tube"
--
local broadcast  = matrix.ext.broadcast
local shuffle    = common.shuffle

-------------------------------------------------------------------------

local data     = preprocess.load_data()
local train    = data.train
local label    = data.train_label
local test     = data.test
local test_ids = data.test_ids

local data_shuffle_rng = random(25482)
local shuffle_rng = random(21349)
local weights_rng = random(82591)
local noise_rng   = random(92581)

local B = 10 -- number of CV blocks
local bunch_size = 128
local h1 = 128
local h2 = 128
local h3 = 0
local h1_dropout = 0.0
local h2_dropout = 0.0
local h3_dropout = 0.0
local lr = 0.01
local mt = 0.9
local wd = 0.01
local mp = 100.0
local mg = 100.0
local min_epochs = 1000
local max_epochs = 8000
local optimizer = "sgd"
local init_w = 0.10

local function renormalize(x)
  broadcast(x.cmul, x, data.label_scale, x)
  broadcast(bind(x.axpy, nil, 1.0), x, data.label_center, x)
  return x
end

local model = ann.mlp.all_all.generate(
  table.concat{
    "%d inputs "%{ data.train:dim(2) },
    h1>0 and ("%d prelu dropout{ prob=#2, random=#1 } "%{ h1 }) or "",
    h2>0 and ("%d prelu dropout{ prob=#3, random=#1 } "%{ h2 }) or "",
    h3>0 and ("%d prelu dropout{ prob=#3, random=#1 } "%{ h3 }) or "",
    "1 linear" },
  { noise_rng, h1_dropout, h2_dropout, h3_dropout })

local cv_mse = ann.loss.mse()
local va_result = matrix(train:dim(1), 1):zeros()
local test_result = matrix(test:dim(1), 1):zeros()

local train_shuffled, label_shuffled, perm = shuffle(data_shuffle_rng, train, label,
                                                     data.train_tube_ids)
local input_ds  = dataset.matrix(train_shuffled)
local output_ds = dataset.matrix(label_shuffled)
local nump   = input_ds:numPatterns()
local blocks = {}
local acc_sizes = { 0 }
do
  local first,last = 1
  for i=1,B do
    last = math.round( (i/B) * nump )
    blocks[i] = {
      input_dataset  = dataset.slice(input_ds, first, last),
      output_dataset = dataset.slice(output_ds, first, last),
    }
    first = last+1
    acc_sizes[#acc_sizes+1] = acc_sizes[#acc_sizes] + blocks[i].input_dataset:numPatterns()
  end
  assert(last == nump)
end

local function train_union(i, field)
  local tbl = {}
  for j=1,B do
    if i~=j then table.insert(tbl, blocks[j][field]) end
  end
  return dataset.union(tbl)
end

for i=1,B do
  print("# ALL: ", train:dim(1), train:dim(2))
  local tr_input_ds   = train_union(i, "input_dataset")
  local tr_output_ds  = train_union(i, "output_dataset")
  local va_input_ds   = blocks[i].input_dataset
  local va_output_ds  = blocks[i].output_dataset
  
  print("# TR: ", tr_input_ds:numPatterns(), tr_input_ds:patternSize())
  print("# VA: ", va_input_ds:numPatterns(), va_input_ds:patternSize())
  
  local trainer = trainable.supervised_trainer(model, ann.loss.mse(), bunch_size,
                                               ann.optimizer[optimizer](),
                                               true, mg)
  trainer:build()
  trainer:randomize_weights{
    name_match = "w.+",
    inf = -init_w,
    sup =  init_w,
    random = weights_rng,
  }
  iterator(trainer:iterate_weights("a.+")):select(2):call("fill", 0.1):apply()
  iterator(trainer:iterate_weights("b.+")):select(2):call("fill", 0.0):apply()

  pcall(trainer.set_option, trainer, "learning_rate", lr)
  pcall(trainer.set_option, trainer, "momentum", mt)
  pcall(trainer.set_layerwise_option, trainer, "w.+", "weight_decay", wd)
  pcall(trainer.set_layerwise_option, trainer, "w.+", "max_norm_penalty", mp)

  print("# MODEL: ", trainer:size(), trainer:get_input_size())

  local pocket = trainable.train_holdout_validation{
    max_epochs = max_epochs,
    min_epochs = min_epochs,
    stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.5),
  }
  
  local train_conf = {
    input_dataset  = tr_input_ds,
    output_dataset = tr_output_ds,
    bunch_size     = bunch_size,
    replacement    = bunch_size,
    shuffle        = shuffle_rng,
  }
  
  local va_conf = {
    input_dataset  = va_input_ds,
    output_dataset = va_output_ds,
    bunch_size     = math.max(bunch_size, 256),
  }
  
  while pocket:execute(function()
      local tr_loss = trainer:train_dataset(train_conf)
      local va_loss = trainer:validate_dataset(va_conf)
      return trainer,math.sqrt(2*tr_loss),math.sqrt(2*va_loss)
                      end) do
    print(pocket:get_state_string(), trainer:norm2("w.*"), trainer:norm2("b.*"))
  end
  
  local best  = pocket:get_state_table().best
  local model = best:get_component()
  
  model:reset()
  local va_hat = renormalize(model:forward(va_input_ds:toMatrix()))
  local va_gt  = renormalize(va_output_ds:toMatrix())
  cv_mse:accum_loss(cv_mse:compute_loss(va_hat, va_gt))
  
  model:reset()
  va_hat:expm1()
  va_result[{ { acc_sizes[i]+1, acc_sizes[i+1] }, ':' }]:axpy(1.0, va_hat)
  
  model:reset()
  local test_hat  = renormalize(model:forward(test)):expm1()
  test_result:axpy(1/B, test_hat)
  
end

local va_loss,va_var = cv_mse:get_accum_loss()
print("# VA LOSS: ", math.sqrt(2*va_loss), math.sqrt(2*va_var))

do
  local inv_perm = matrixInt32(perm:size())
  for i=1,#perm do inv_perm[perm[i]] = i end
  local va_df = data_frame{
    data = { id=matrixInt32(va_result:dim(1)):linspace(),
             cost=va_result:index(1,inv_perm) },
    columns = { "id", "cost" },
  }
  va_df:to_csv("validation_result.csv")
end

do
  local test_df = data_frame{
    data = { id=test_ids,
             cost=test_result },
    columns = { "id", "cost" },
  }
  test_df:to_csv("test_result.csv")
end
