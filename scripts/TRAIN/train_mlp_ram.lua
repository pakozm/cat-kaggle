april_print_script_header(arg)
package.path = arg[0]:dirname() .. "/../?.lua;" .. package.path
local common     = require "common"
local preprocess = require "PREPROCESS.load_train_and_tube"
--
local broadcast  = matrix.ext.broadcast
local shuffle    = common.shuffle

-------------------------------------------------------------------------

local data
do
  local DATA_FILENAME = "/tmp/data_ram.mat.gz"
  if io.open(DATA_FILENAME) then
    data = util.deserialize(DATA_FILENAME)
    print("# DATA RETRIEVED FROM DISK")
  else
    data = preprocess.load_data_ram()
    util.serialize(data, DATA_FILENAME)
  end
end

local train    = data.train
local label    = data.train_label
local test     = data.test
local test_ids = data.test_ids

-- train:toTabFilename("/tmp/ram.mat.gz")

local data_shuffle_seed = tonumber(arg[4] or 53968)
local shuffle_seed = tonumber(arg[5] or 4852869)
local weights_seed = tonumber(arg[6] or 5968)
local noise_seed = tonumber(arg[7] or 72659)

local data_shuffle_rng = random(data_shuffle_seed) --random(25482)
local shuffle_rng = random(shuffle_seed)    --random(21349)
local weights_rng = random(weights_seed)       -- random(82591)
local noise_rng   = random(noise_seed)     -- random(92581)

local B = tonumber(10) -- number of CV blocks
local E = tonumber(10) -- CV early stopping
local bunch_size = tonumber(128)
local replacement = 2^13
local h1 = tonumber(arg[1] or 128)
local h2 = tonumber(arg[2] or 64)
local h3 = tonumber(arg[3] or 256)
local input_dropout = tonumber(0.0)
local h1_dropout = tonumber(0.1)
local h2_dropout = tonumber(0.1)
local h3_dropout = tonumber(0.1)
local lr = tonumber(1.0) -- 1.0 -- 0.01 -- 0.1
local wd = tonumber(0.0001235523855)
local mp = tonumber(100.0)
local mg = tonumber(100.0)
local mt = tonumber(0.0)
local mt_decay = tonumber(1.00)
local max_mt   = tonumber(0.99)
local min_epochs = tonumber(40)
local max_epochs = tonumber(600)
local optimizer = "adadelta"
local init_w = tonumber(1.1)

assert(E <= B)

local function renormalize(x)
  broadcast(x.cmul, x, data.label_scale, x)
  broadcast(bind(x.axpy, nil, 1.0), x, data.label_center, x)
  return x
end

local function generate_model()
  -- The following function builds a component which joins all input matrices,
  -- so categorical data is projected into sqrt(one-hot-vector-size) layers and
  -- numerical data is by-passed to the next layer.
  local function build_input_join_component()
    local isize = 0
    local join = ann.components.join{ name="input_join" }
    local name2id = table.invert(data.train_s_list)
    local cat_sizes = table.imap(data.train_s_cats, lambda'|x|#x')
    cat_sizes[0] = 1 -- sentinel
    -- process all data at in the given order
    for k,fields in ipairs(data.result_order) do
      local size = 0 -- number of columns needed for the fields list
      for _,field in ipairs(fields) do
        size = size + cat_sizes[name2id[field] or 0]
      end
      print("#",size,table.concat(fields, ","))
      local wname -- set with a string when categorical data
      local bname -- only used for component fields
      if fields[1]:find("^cmp_") then
        -- all components share same bias and weights
        wname = "w1_cat_cmp_" .. k
        bname = "b1_cat_cmp_" .. k
      elseif fields[1]:find("^qcmp_") then
        -- all components share same bias and weights
        wname = "w1_cat_qcmp_" .. k
        bname = "b1_cat_qcmp_" .. k
      elseif fields[1]:find("^spc_") then
        -- all specs share same weights matrix
        wname = "w1_cat_spec_" .. k
        bname = "b1_cat_spec_" .. k
      elseif #fields==1 and name2id[fields[1]] then
        -- other categorical data has different weights matrix
        wname = "w1_cat_"..k
      end
      if wname then -- it is not nil for categorical data
        local code_size = math.max(2, math.floor(math.sqrt(size)))
        print("# CODE ", code_size)
        if bname then
          join:add( ann.components.hyperplane{
                      input  = size,
                      output = code_size,
                      --
                      name    = "layer1_"..k,
                      dot_product_weights = wname,
                      bias_weights        = bname,
          } )
        else
          join:add( ann.components.dot_product{
                      input  = size,
                      output = code_size,
                      --
                      name    = "layer1_"..k,
                      weights = wname,
          } )
        end
        isize = isize + code_size
      else
        -- non categorical data
        join:add( ann.components.base{ name="layer1_"..k, size=size } )
        isize = isize + size
      end
    end
    return join,isize
  end
  -----------------------------------------------------------------------------
  local model      = ann.components.stack{ name="CAT_ANN" }
  local join,isize = build_input_join_component()
  model:push( join )
  model:push( ann.mlp.all_all.generate(
                table.concat{
                  "%d inputs "%{ isize },
                  "dropout{ prob=#2, random=#1 }",
                  h1>0 and ("%d prelu dropout{ prob=#3, random=#1 } "%{ h1 }) or "",
                  h2>0 and ("%d prelu dropout{ prob=#4, random=#1 } "%{ h2 }) or "",
                  h3>0 and ("%d prelu dropout{ prob=#5, random=#1 } "%{ h3 }) or "",
                  "1 linear" },
                { noise_rng, input_dropout, h1_dropout, h2_dropout, h3_dropout, first_count=2 }):
                unroll() )
  return model
end

local model  = generate_model()
local cv_mse = ann.loss.mse()
local va_result   = matrix(train:dim(1), 1):zeros()
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

local _=1
local vars={}
while true do
  local key,value = debug.getlocal(1,_)
  _=_+1
  if not key then break end
  if type(value) == "string" or type(value) == "number" then
    if key ~= "_" then table.insert(vars,{key,value}) end
  end
end
table.sort(vars,lambda"|a,b|a[1]<b[1]")
print("# PARAMETRIZATION")
local str = iterator.zip(iterator.duplicate("#"),
                         iterator(vars):map(table.unpack)):concat(" ","\n")
print(str)
print("################")
local md5_in,md5_out = io.popen2("md5sum")
md5_in:write(str)
md5_in:close()
local md5 = md5_out:read("*a"):match("^([^%s]+).*$")
print("# MD5", md5)

for i=1,E do
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
    use_fanin  = true,
    use_fanout = true,
  }
  iterator(trainer:iterate_weights("a.+")):select(2):call("fill", 0.1):apply()
  iterator(trainer:iterate_weights("b.+")):select(2):call("fill", 0.01):apply()
  
  pcall(trainer.set_option, trainer, "learning_rate", lr)
  pcall(trainer.set_option, trainer, "momentum", mt)
  pcall(trainer.set_layerwise_option, trainer, "w.+", "weight_decay", wd)
  pcall(trainer.set_layerwise_option, trainer, "w.+", "max_norm_penalty", mp)
  pcall(trainer.set_layerwise_option, trainer, ".+cat.+", "weight_decay", 0.0)

  print("# MODEL: ", trainer:size(), trainer:get_input_size())
  --  util.serialize(util.to_lua_string({shuffle_rng,trainer,perm}), "/tmp/network2.lua")
  
  local pocket = trainable.train_holdout_validation{
    max_epochs = max_epochs,
    min_epochs = min_epochs,
    stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2.0),
  }
  
  local train_conf = {
    input_dataset  = tr_input_ds,
    output_dataset = tr_output_ds,
    bunch_size     = bunch_size,
    replacement    = replacement,
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
    pcall(trainer.set_option, trainer, "momentum",
          math.min(max_mt, mt*(mt_decay^pocket:get_state_table().current_epoch)))
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
  test_result:axpy(1/E, test_hat)

  local va_loss,va_var = cv_mse:get_accum_loss()
  print("# partial", math.sqrt(2*va_loss), math.sqrt(2*va_var))
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
  va_df:to_csv("RAM_MLPS/val_stage0_%s.VAL%s.csv"%{md5,tostring(math.sqrt(2*va_loss))})
end

do
  local test_df = data_frame{
    data = { id=test_ids,
             cost=test_result },
    columns = { "id", "cost" },
  }
  test_df:to_csv("RAM_MLPS/test_stage0_%s.VAL%s.csv"%{md5,tostring(math.sqrt(2*va_loss))})
end
