package.path = arg[0]:dirname() .. "/../?.lua;" .. package.path
local common     = require "common"
local preprocess = require "PREPROCESS.load_train_and_tube"
--
local shuffle    = common.shuffle
-------------------------------------------------------------------------
local data     = preprocess.load_data()
local train    = data.train
local label    = data.train_label

local data_shuffle_rng = random(25482)
local B = 10 -- number of CV blocks
local train_shuffled, label_shuffled, perm = shuffle(data_shuffle_rng, train, label,
                                                     data.train_tube_ids)
local nump = #data.train_tube_ids
local blocks = {}
local acc_sizes = { 0 }
local inv_perm = matrixInt32(perm:size())
for i=1,#perm do inv_perm[perm[i]] = i end
local first,last = 1
for i=1,B do
  last = math.round( (i/B) * nump )
  blocks[i] = {}
  for j=first,last do table.insert(blocks[i], perm[j]) end
  first = last+1
  acc_sizes[#acc_sizes+1] = acc_sizes[#acc_sizes] + #blocks[i]
end
assert(last == nump)

print(iterator(blocks):map(table.unpack):concat(" ","\n"))
