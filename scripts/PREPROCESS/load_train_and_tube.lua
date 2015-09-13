local DATE_DELTA=5
local K=10

local function filter(df, field, filtered)
  local col = df[{field}]
  local inv_filtered = table.invert(filtered)
  for i=1,#col do
    if not inv_filtered[col[i]] then col[i] = 9999 end
  end
end

local function count(train, field, dict)
  local m,dict = train:as_matrix(field,{dtype="categorical",categories={dict}})
  local dict   = dict[1]
  local params = {symbols=dict and iterator.range(#dict):table()}
  return stats.ihist(m,params):select(2,3),dict
end

local function take_k(k, counts, dict)
  local counts   = counts:gt(k):to_index():flatten()
  local filtered = {}
  for i=1,#counts do filtered[i] = dict[counts[i]] end
  if #filtered ~= #dict then table.insert(filtered, 9999) end
  return filtered
end

local function remove_rare_events(train, test, fields, k)
  if type(fields) ~= "table" then fields = {fields} end
  print("# Filtering categorical data: " .. table.concat(fields, " "))
  local names = set()
  for i,field in ipairs(fields) do names:update(set(train:levels(field))) end
  local dict = names:keys()
  local counts = count(train, fields[1], dict)
  for i=2,#fields do
    local field = fields[i]
    local aux_counts = count(train, field, dict)
    counts:axpy(1, aux_counts)
  end
  local filtered = take_k(k, counts, dict)
  for i,field in ipairs(fields) do
    filter(train, field, filtered)
    if test then filter(test, field, filtered) end
  end
  return filtered
end

-- checks the correctness of the given matrix
local function check(m)
  assert(m)
  assert(m:eq(nan):count_ones() == 0)
  return m
end

-- given a date string returns a table with date values
local function date_parser(str)
  local year,month,day = str:match("^(.+)%-(.+)%-(.+)$")
  return {
    year  = tonumber(year),
    month = tonumber(month),
    day   = tonumber(day),
  }
end

-- given a date string returns a table with date values
local function date_parser2(str)
  local day,month,year = str:match("^(.+)%-(.+)%-(.+)$")
  return {
    year  = tonumber(year),
    month = tonumber(month),
    day   = tonumber(day),
  }
end

local function date_parser3(str)
  local year,month,day = str:match("^(.+)%-(.+)%-(.+) .+$")
  return {
    year  = tonumber(year),
    month = tonumber(month),
    day   = tonumber(day),
  }
end

-- Receives a data_frame and the date field and splits it into year, month, day
-- and dow (day-of-week). At the end, it DROPS the date field.
local function split_date(df, field, _date_parser_)
  -- auxiliary columns
  df[{"timestamp"}] = df:parse_datetime(field,  _date_parser_ or date_parser)
  df[{"datetime"}]  = df:map("timestamp", bind(os.date, "*t"))
  -- date columns
  df[{"year"}]  = df:map("datetime", function(x) return math.floor(x.year/DATE_DELTA)*DATE_DELTA end)
  df[{"month"}] = df:map("datetime", function(x) return x.month end)
  df[{"day"}]   = df:map("datetime", function(x) return x.day end)
  df[{"dow"}]   = df:map("datetime", function(x) return x.wday end)
  -- drop all auxiliary columns
  df:drop(2, "timestamp", "datetime", field)
end

local function transform(df, n_list, s_list, d_list,
                         categories_s, categories_d)
  local n = df:as_matrix(table.unpack(n_list)):clamp(0.0,math.huge):log1p()
  local s,categories_s = df:as_matrix(multiple_unpack(s_list, { { dtype="categorical", categorical_dtype="sparse", NA="NONE",
                                                                  categories = categories_s } }))
  local d,categories_d = df:as_matrix(multiple_unpack(d_list, { { dtype="categorical", NA="NONE",
                                                                  categories = categories_d } }))
  return n,s,categories_s,d,categories_d 
end

local function transform2(df, n_list, s_list, d_list,
                          categories_s, categories_d)
  local categories_d = categories_d or {}
  local matrices = {}
  for _,field in ipairs(n_list) do
    print("# Transforming", field)
    matrices[field] = df:as_matrix(field):clamp(0.0,math.huge):log1p()
  end
  for i,field in ipairs(s_list) do
    print("# Transforming", field)
    matrices[field],categories_s[i] = df:as_matrix(field, { dtype="categorical", categorical_dtype="sparse", NA="NONE",
                                                            categories = {categories_s[i]} })
    categories_s[i] = categories_s[i][1]
    matrices[field] = matrices[field]:to_dense()
  end
  for i,field in ipairs(d_list) do
    print("# Transforming", field)
    matrices[field],categories_d[i] = df:as_matrix(field, { dtype="categorical", NA="NONE",
                                                            categories = {categories_d[i]} })
    categories_d[i] = categories_d[i][1]
    assert(matrices[field]:min() == 0)
    assert(matrices[field]:max() == 1)
  end
  return matrices,categories_s,categories_d 
end

local function transform3(df, n_log_list, n_list, s_list, d_list,
                          categories_s, categories_d)
  local categories_d = categories_d or {}
  local matrices = {}
  for _,field in ipairs(n_log_list) do
    print("# Transforming", field)
    matrices[field] = df:as_matrix(field):clamp(0.0,math.huge):log1p()
  end
  for _,field in ipairs(n_list) do
    print("# Transforming", field)
    matrices[field] = df:as_matrix(field):clamp(0.0,math.huge)
  end
  for i,field in ipairs(s_list) do
    print("# Transforming", field)
    matrices[field],categories_s[i] = df:as_matrix(field, { dtype="categorical", categorical_dtype="sparse", NA="NONE",
                                                            categories = {categories_s[i]} })
    categories_s[i] = categories_s[i][1]
    matrices[field] = matrices[field]:to_dense()
  end
  for i,field in ipairs(d_list) do
    print("# Transforming", field)
    matrices[field],categories_d[i] = df:as_matrix(field, { dtype="categorical", NA="NONE",
                                                            categories = {categories_d[i]} })
    categories_d[i] = categories_d[i][1]
    assert(matrices[field]:min() == 0)
    assert(matrices[field]:max() == 1)
  end
  return matrices,categories_s,categories_d 
end

local function build_input_matrix(ids, data_joined,
                                  tube_ids, tube_data_joined)
  local tube_reorder_idx = matrixInt32(data_joined:dim(1))
  local inv_tube_ids = table.invert(tube_ids)
  for i,id in ipairs(ids) do tube_reorder_idx[i] = inv_tube_ids[id] end
  local tube_reordered_data = tube_data_joined:index(1, tube_reorder_idx)
  -- concatenate all the data
  return matrix.join(2, data_joined, tube_reordered_data)
end

local function bag_of_symbols(df, dict, fields, prefix, values)
  local new_fields,names = {},{}
  print("# Bag-of-symbols", fields[1])
  for _,symbol in ipairs(dict) do
    local name = prefix .. tostring(symbol)
    print("#", symbol, "=>", name)
    table.insert(new_fields, name)
    df[{name}] = matrix(df:nrows()):zeros()
    names[symbol] = name
  end
  if not values then
    for j,field in ipairs(fields) do
      for k,x in ipairs(df[{field}]) do
        local name = assert(names[x])
        df[{name}][k] = 1
      end
    end
  else
    for j,field in ipairs(fields) do
      for k,x in ipairs(df[{field}]) do
        local name = assert(names[x])
        df[{name}][k] = df[{name}][k] + df[{values[j]}][k]
      end
    end
  end
  return new_fields
end

local function apply_tf_idf(train, test, names)
  for i,name in ipairs(names) do
    print("# Applying TF-IDF", name)
    local tr  = train[name]
    local N   = #tr
    local df  = tr:gt(0.0):count_ones()
    april_assert(df > 0, "Found zero df for name %s", name)
    local idf = math.log( N / df )
    tr:scal( idf )
    test[name]:scal( idf )
  end
end

-------------------------------------------------------------------------

local function load_data(path)
  local path = path or "../input"
  -- tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,bracket_pricing,quantity,cost
  -- TA-00002,S-0066,2013-07-07,0,0,Yes,1,21.9059330191461
  local train = data_frame.from_csv(path .. "/train_set.csv", { NA="NONE" })
  -- id,tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,bracket_pricing,quantity
  local test  = data_frame.from_csv(path .. "/test_set.csv", { NA="NONE", index="id" })
  -- tube_assembly_id,material_id,diameter,wall,length,num_bends,bend_radius,end_a_1x,end_a_2x,end_x_1x,end_x_2x,end_a,end_x,num_boss,num_bracket,other
  -- TA-00001,SP-0035,12.7,1.65,164,5,38.1,N,N,N,N,EF-003,EF-003,0,0,0
  local tube  = data_frame.from_csv(path .. "/tube.csv", { NA="NONE" })
  -- tube_assembly_id,component_id_1,quantity_1,component_id_2,quantity_2,component_id_3,quantity_3,component_id_4,quantity_4,component_id_5,quantity_5,component_id_6,quantity_6,component_id_7,quantity_7,component_id_8,quantity_8
  -- TA-00001,C-1622,2,C-1629,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA
  -- local bill_of_materials = data_frame.from_csv(path .. "/bill_of_materials.csv", { NA="NONE" })
  -------------------------------------------------------------------------

  split_date(train, "quote_date")
  split_date(test,  "quote_date")

  local train = train:merge(tube, { on="tube_assembly_id" })
  local test  = test:merge(tube,  { on="tube_assembly_id" })
  
  remove_rare_events(train, test, "supplier", K)
  remove_rare_events(train, test, "material_id", K)
  remove_rare_events(train, test, "end_a", K)
  remove_rare_events(train, test, "end_x", K)

  -- train:to_csv("/tmp/jarl.csv")

  -- numerical data
  local train_n_list = { "annual_usage", "min_order_quantity", "quantity",
                         "diameter", "wall", "length", "num_bends", "bend_radius" }
  -- categorical sparse
  local train_s_list = { "supplier", "year", "month", "day", "dow",
                         "material_id", "end_a", "end_x"}
  -- categorical dense
  local train_d_list = { "bracket_pricing",
                         "end_a_1x", "end_a_2x", "end_x_1x", "end_x_2x"}
  
  local train_n,train_s,train_s_cats,train_d,train_d_cats = transform(train,
                                                                      train_n_list,
                                                                      train_s_list,
                                                                      train_d_list,
                                                                      { nil,
                                                                        iterator.range(1980,2015,DATE_DELTA):table(),
                                                                        iterator.range(12):table(),
                                                                        iterator.range(31):table(),
                                                                        iterator.range(7):table(), })
  
  local test_n,test_s,_,test_d,_ = transform(test,
                                             train_n_list,
                                             train_s_list,
                                             train_d_list,
                                             train_s_cats,
                                             train_d_cats)
  
  local train_n,center,scale = stats.standardize(train_n)
  local test_n = stats.standardize(test_n, { center=center, scale=scale })
  
  local train_costs,cost_center,cost_scale =
    stats.standardize( check( train:as_matrix("cost"):log1p() ), {
                         -- center = matrix(1,1):fill(0),
                         -- scale  = matrix(1,1):fill(1),
    })
  
  local train_data_joined = matrix.join(2, train_n, train_s:to_dense(), train_d)
  local test_data_joined  = matrix.join(2, test_n, test_s:to_dense(), test_d)

  local train_tube_ids = train[{"tube_assembly_id"}]
  local test_tube_ids  = test[{"tube_assembly_id"}]
  local test_ids       = test[{"id"}]
  
  return { train = train_data_joined,
           train_label = train_costs,
           test = test_data_joined,
           test_ids = test_ids,
           train_tube_ids = train_tube_ids,
           label_center = cost_center,
           label_scale = cost_scale, }
end

local function load_data_ram()
  local path = path or "../input/ram"
  local train = data_frame.from_csv(path .. "/Train_features_final.csv", { NA="NONE" })
  print("# Train data loaded")
  local test  = data_frame.from_csv(path .. "/Test_features_final.csv", { NA="NONE", index="id" })
  print("# Test data loaded")
  
  -- "","tube_assembly_id","supplier","quote_date","annual_usage",
  -- "min_order_quantity","bracket_pricing","quantity","cost",
  -- "component_id_1","weight_1","quantity_1",
  -- "component_id_2","weight_2","quantity_2",
  -- "component_id_3","weight_3","quantity_3",
  -- "component_id_4","weight_4","quantity_4",
  -- "component_id_5","weight_5","quantity_5",
  -- "component_id_6","weight_6","quantity_6",
  -- "component_id_7","weight_7","quantity_7",
  -- "component_id_8","weight_8","quantity_8",
  -- "spec1","spec2","spec3","spec4","spec5","spec6","spec7","spec8","spec9","spec10",
  -- "material_id","diameter","wall","length","num_bends","bend_radius",
  -- "end_a_1x","end_a_2x","end_x_1x","end_x_2x","end_category","end_a","end_x",
  -- "num_boss","num_bracket","other","days_elapsed",
  -- "count_of_components","count_of_specs",
  -- "inner_volume","outer_volume","material_volume","surface_area",
  -- "cost_log","cost_log_est","cost_est",
  -- "body","bushing","cap","clamp","cleat","clip","collar","connector","cover",
  -- "disc","elbow","filler","flange","hanger_coat","hasp","head","hook","hose",
  -- "housing","joint","link","lug","manifold","nipple","nozzle","orifice",
  -- "pin","pipe","plate","plug","post","reducer","requirements","retainer",
  -- "ring","rod","saddle","screen","screw","seal","sheet","shell","shroud",
  -- "sleeve","socket","spacer","stem","strap","tee","tube","union","valve",
  -- "washer","adapter","angle","baffle","bar","bellow","block","other_comp",
  -- "welded","bolt_nut","fitted","brazed","coupled","cast","riveted","studded",
  -- "hydraulic","inflation","orientation","unique","groove","plating","blind_hole",
  -- "num_dim","assembly_weight","crude_oil","sdr","end_check","num_bbbo",
  -- "distinct_components","bin_count_comp","bin_count_specs",
  -- "bin_dist_comp","bin_moq","num_end_form"

  local numerical_asymmetric_data_fields = {
    "weight_1",
    "weight_2",
    "weight_3",
    "weight_4",
    "weight_5",
    "weight_6",
    "weight_7",
    "weight_8",
    "inflation","assembly_weight",
  }
  local numerical_log_gaussian_data_fields = {
    "length",
    "inner_volume","outer_volume","material_volume","surface_area",
    "sdr",
  }
  local numerical_large_data_fields = {
    "diameter","wall",
  }
  local numerical_other_data_fields = {
    "days_elapsed",
  }
  local integer_data_fields = {
    "count_of_components","count_of_specs",
    "num_bends","num_boss","num_bracket","other","num_dim",
    "num_bbbo","num_end_for",
    "annual_usage","min_order_quantity","quantity",
    "quantity_1",
    "quantity_2",
    "quantity_3",  
    "quantity_4",
    "quantity_5",
    "quantity_6",
    "quantity_7",
    "quantity_8",
    "distinct_components",
    "orientation",
  }
  local categorical_data_fields = {
    "supplier",
    "material_id",
    "end_a","end_x",
    "elbow",
    "hanger_coat",
    "seal","sleeve","socket",
    "other_comp","bolt_nut",
    "fitted",
    "unique","groove",
    "bin_count_specs","bin_moq",
  }
  -- local shared_categorical_data_fields = {
  --   "component_id_1","component_id_2","component_id_3","component_id_4",
  --   "component_id_5","component_id_6","component_id_7","component_id_8",
  --   "spec1","spec2","spec3","spec4","spec5","spec6","spec7","spec8","spec9","spec10",
  -- }
  local quantized_data_fields = {
    "bend_radius",
  }
  local binary_data_fields = {
    "bracket_pricing",
    "end_a_1x","end_a_2x","end_x_1x","end_x_2x",
    "end_check",
    "cover",
    "hook",
    "bin_count_comp","bin_dist_comp",
  }
  local ignored_data_fields = {
    "tube_assembly_id",
    "end_category","cost_log","cost_log_est","cost_est",
    "connector","disc","filler",
    "flange","head","hose","joint","nozzle",
    "requirements","retainer","ring","rod","saddle","screen","screw",
    "sheet","shroud","spacer","strap",
    "adapter",
    "angle","baffle","bar","bellow","block","coupled",
    "studded","blind_hole",
    "bushing",
    "clamp","cleat","clip","cast","collar",
    "valve","washer","tee","union","tube","welded",
    "stem","shell","riveted","post",
    "plating",
    "plate","pipe","pin",
    "orifice","nipple",
    "manifold","link","lug","housing",
    "hasp","cap","brazed",
    "body",
    "plug","reducer","hydraulic",
  }

  local specs = {"spec1","spec2","spec3","spec4","spec5",
                 "spec6","spec7","spec8","spec9","spec10",}
  local quantities = {"quantity_1","quantity_2",
                      "quantity_3","quantity_4",
                      "quantity_5","quantity_6",
                      "quantity_7","quantity_8",}
  
  local components = {"component_id_1","component_id_2",
                      "component_id_3","component_id_4",
                      "component_id_5","component_id_6",
                      "component_id_7","component_id_8",}

  local components_dict = remove_rare_events(train, test, components, K)
  local specs_dict      = remove_rare_events(train, test, specs, K)

  for _,field in ipairs(categorical_data_fields) do
    remove_rare_events(train, test, field, K)
  end
  
  split_date(train, "quote_date", date_parser2)
  split_date(test,  "quote_date", date_parser2)

  print("# Rare events removed and date split")

  -- convert components and quantities into bag of symbols
  local bag_of_components = bag_of_symbols(train, components_dict, components, "cmp_")
  local bag_of_quantities = bag_of_symbols(train, components_dict, components, "qcmp_", quantities)
  local bag_of_specs      = bag_of_symbols(train, specs_dict, specs, "spc_")
  
  bag_of_symbols(test, components_dict, components, "cmp_")
  bag_of_symbols(test, components_dict, components, "qcmp_", quantities)
  bag_of_symbols(test, specs_dict, specs, "spc_")
  
  -- remove the not available category
  bag_of_components = iterator(bag_of_components):filter(lambda'|x|x~="cmp_not applicable"'):table()
  bag_of_quantities = iterator(bag_of_quantities):filter(lambda'|x|x~="qcmp_not applicable"'):table()
  bag_of_specs      = iterator(bag_of_specs):filter(lambda'|x|x~="spc_not applicable"'):table()
  
  train:to_csv("/tmp/jarl.csv")

  -- numerical data
  local train_n_log_list = set(numerical_asymmetric_data_fields):
    update(set(numerical_log_gaussian_data_fields),
           set(numerical_large_data_fields),
           set(numerical_other_data_fields),
           set(integer_data_fields),
           set(bag_of_quantities)):keys()
  local train_n_sqrt_list = {}
  local train_n_list = {}
  
  -- categorical sparse
  local train_s_list = categorical_data_fields
  local train_s_list = table.join({ "year", "month", "day", "dow" },train_s_list)
  
  -- categorical dense
  local train_d_list = table.join(binary_data_fields, bag_of_specs)

  local result_order = {
    bag_of_quantities,
    --
    bag_of_specs,
    --
    {"supplier"},
    {"material_id"},
    {"end_a"},
    {"end_x"},
    {"elbow"},
    {"hanger_coat"},
    {"seal"},
    {"sleeve"},
    {"socket"},
    {"other_comp"},
    {"bolt_nut"},
    {"fitted"},
    {"unique"},
    {"groove"},
    {"bin_count_specs"},
    {"bin_moq"},
    {"year"},
    {"month"},
    {"day"},
    {"dow"},
    {
      --
      "inflation","assembly_weight",
      --
      "length",
      "inner_volume","outer_volume","material_volume","surface_area",
      "sdr",
      --
      "diameter","wall",
      --
      "days_elapsed",
      --
      "count_of_components","count_of_specs",
      "num_bends","num_boss","num_bracket","other","num_dim",
      "num_bbbo","num_end_for",
      "annual_usage","min_order_quantity","quantity",
      "distinct_components",
      "orientation",
      --
      "bracket_pricing",
      "end_a_1x","end_a_2x","end_x_1x","end_x_2x",
      "end_check",
      "cover",
      "hook",
      "bin_count_comp","bin_dist_comp",
    }
  }

  local tr_matrices,train_s_cats,train_d_cats = transform2(
    train,
    train_n_log_list,
    train_s_list,
    train_d_list,
    { iterator.range(1980,2015,DATE_DELTA):map(lambda'|x|x%100'):table(),
      iterator.range(12):table(),
      iterator.range(31):table(),
      iterator.range(7):table(), })
  
  local te_matrices = transform2(test,
                                 train_n_log_list,
                                 train_s_list,
                                 train_d_list,
                                 train_s_cats,
                                 train_d_cats)

  for _,field in pairs(train_n_log_list) do
    print("# Standarizing", field)
    local center,scale
    do
      scale,center = stats.std(tr_matrices[field])
      tr_matrices[field]:scalar_add(-center):scal(1/scale)
    end
    do
      te_matrices[field]:scalar_add(-center):scal(1/scale)
    end
  end

  print("# Data standarized")

  local costs_log1p = train:as_matrix("cost"):log1p()
  local train_costs,cost_center,cost_scale =
    stats.standardize( check( costs_log1p ), {
                         center = matrix(1,1,{stats.percentile(costs_log1p, 0.5)}), -- matrix(1,1):fill(0),
                         scale  = matrix(1,1):fill(1),
    })

  local tr_matrices_array,te_matrices_array = {},{}
  for _,fields in ipairs(result_order) do
    for _,field in ipairs(fields) do
      table.insert(tr_matrices_array, (assert(tr_matrices[field],
                                              "Not found train field "..field)))
      table.insert(te_matrices_array, (assert(te_matrices[field],
                                              "Not found test field "..field)))
    end
  end
  
  local train_data_joined = matrix.join(2, tr_matrices_array)
  local test_data_joined  = matrix.join(2, te_matrices_array)

  local train_tube_ids = train[{"tube_assembly_id"}]
  local test_tube_ids  = test[{"tube_assembly_id"}]
  local test_ids       = test[{"id"}]

  print("# Data preprocessed")
  
  return { train = train_data_joined,
           train_label = train_costs,
           test = test_data_joined,
           test_ids = test_ids,
           train_tube_ids = train_tube_ids,
           label_center = cost_center,
           label_scale = cost_scale,
           train_n_log_list = train_n_log_list,
           train_s_list = train_s_list,
           train_d_list = train_d_list,
           train_s_cats = train_s_cats,
           train_d_cats = train_d_cats,
           result_order = result_order,
  }
end

local function load_data_team()
  local path = path or "../input/team"
  local train = data_frame.from_csv(path .. "/TRAIN.csv", { NA="NONE" })
  print("# Train data loaded")
  local test  = data_frame.from_csv(path .. "/TEST.csv", { NA="NONE", index="id" })
  print("# Test data loaded")
  
  -- ,tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,
  -- bracket_pricing,quantity,cost,material_id,diameter,wall,length,num_bends,
  -- bend_radius,end_a_1x,end_a_2x,end_x_1x,end_x_2x,end_a,end_x,num_boss,
  -- num_bracket,other,s_mean,s_median,s_sd,m_mean,m_median,m_sd,component_id_1,
  -- quantity_1,component_id_2,quantity_2,component_id_3,quantity_3,component_id_4,
  -- quantity_4,component_id_5,quantity_5,component_id_6,quantity_6,component_id_7,
  -- quantity_7,component_id_8,quantity_8,weight,num_unique_features,
  -- num_orientations,connection_type_counts,interactions,num_comp,num_spec,
  -- SP.0063,SP.0012,SP.0080,SP.0007,SP.0026,SP.0082,SP.0069,SP.0070

  local numerical_data_fields = {
    "s_mean",
    "s_median",
    "s_sd",
    "m_mean",
    "m_median",
    "m_sd",
    "quantity_1",
    "quantity_2",
    "quantity_3",
    "quantity_4",
    "quantity_5",
    "quantity_6",
    "quantity_7",
    "quantity_8",
  }
  local numerical_log_gaussian_data_fields = {
    "annual_usage",
    "num_boss",
    "num_bracket",
    "other",
    "num_unique_features",
    "num_orientations",
    "connection_type_counts",
    "num_comp",
    "num_spec",
    --
    "min_order_quantity",
    "quantity",
    "diameter",
    "wall",
    "length",
    "num_bends",
    "bend_radius",
    "weight",
    "interactions",
  }
  local categorical_data_fields = {
    "supplier",
    "material_id",
    "end_x",
  }
  local shared_categorical_data_fields = {
    "component_id_1","component_id_2","component_id_3","component_id_4",
    "component_id_5","component_id_6","component_id_7","component_id_8",
  }
  local binary_data_fields = {
    "bracket_pricing",
    "end_a_1x",
    "end_a_2x",
    "end_x_2x",
    "end_a",
    "SP.0063",
    "SP.0012",
    "SP.0080",
    "SP.0007",
    "SP.0026",
    "SP.0082",
    "SP.0069",
    "SP.0070",  
  }
  local ignored_data_fields = {
  }

  local components_dict = remove_rare_events(train, test,
                                             shared_categorical_data_fields, K)
  
  for _,field in ipairs(categorical_data_fields) do
    remove_rare_events(train, test, field, K)
  end
  
  split_date(train, "quote_date", date_parser3)
  split_date(test,  "quote_date", date_parser3)
  
  print("# Rare events removed and date split")

  -- convert components and quantities into bag of symbols
  local quantities = {"quantity_1","quantity_2",
                      "quantity_3","quantity_4",
                      "quantity_5","quantity_6",
                      "quantity_7","quantity_8",}
  local bag_of_components = bag_of_symbols(train, components_dict,
                                           shared_categorical_data_fields, "cmp")
  local bag_of_quantities = bag_of_symbols(train, components_dict,
                                           shared_categorical_data_fields, "qcmp",
                                           quantities)
  bag_of_symbols(test, components_dict,
                 shared_categorical_data_fields, "cmp")
  bag_of_symbols(test, components_dict,
                 shared_categorical_data_fields, "qcmp", quantities)

  -- remove the not available category
  bag_of_components = iterator(bag_of_components):filter(lambda'|x|x~="cmp0"'):table()
  bag_of_quantities = iterator(bag_of_quantities):filter(lambda'|x|x~="qcmp0"'):table()
  
  train:to_csv("/tmp/jarl.csv")

  -- numerical data
  local train_n_log_list = table.join(numerical_log_gaussian_data_fields, bag_of_quantities)
  local train_n_list = numerical_data_fields
  
  -- categorical sparse
  local train_s_list = categorical_data_fields
  -- table.join(shared_categorical_data_fields,
  -- categorical_data_fields)
  local train_s_list = table.join({ "year", "month", "day", "dow" },train_s_list)
  
  -- categorical dense
  local train_d_list = table.join(binary_data_fields, bag_of_components)

  local result_order = {
    --
    {"supplier"},
    {"material_id"},
    {"end_x"},
    -- bag_of_components,
    bag_of_quantities,
    {
      "SP.0063",
      "SP.0012",
      "SP.0080",
      "SP.0007",
      "SP.0026",
      "SP.0082",
      "SP.0069",
      "SP.0070",
    },
    {"year"},
    {"month"},
    {"day"},
    {"dow"},
    {
      "annual_usage",
      "num_boss",
      "num_bracket",
      "other",
      "s_mean",
      "s_median",
      "s_sd",
      "m_mean",
      "m_median",
      "m_sd",
      "num_unique_features",
      "num_orientations",
      "connection_type_counts",
      "num_comp",
      "num_spec",
    },
    {
      "bracket_pricing",
      "end_a_1x",
      "end_a_2x",
      "end_x_2x",
      "end_a",
    }
  }

  local tr_matrices,train_s_cats,train_d_cats = transform3(
    train,
    train_n_log_list,
    train_n_list,
    train_s_list,
    train_d_list,
    { iterator.range(1980,2015,DATE_DELTA):table(),
      iterator.range(12):table(),
      iterator.range(31):table(),
      iterator.range(7):table(), })
  
  local te_matrices = transform3(test,
                                 train_n_log_list,
                                 train_n_list,
                                 train_s_list,
                                 train_d_list,
                                 train_s_cats,
                                 train_d_cats)

  -- apply_tf_idf(tr_matrices, te_matrices, bag_of_quantities)
  if false then
    for _,field in pairs(table.join(train_n_log_list, train_n_list)) do
      -- if not field:find("^qcmp") then
      print("# Standarizing", field)
      local center,scale
      center = matrix(1,1):fill(0)
      -- scale  = matrix(1,1):fill(1)
      do
        -- center = matrix(1,1,{(tr_matrices[field]:min())})
        -- scale  = matrix(1,1,{(tr_matrices[field]:max())}) - center
        local tr_std_m
        tr_matrices[field],center,scale = stats.standardize(tr_matrices[field],
                                                            {center=center,scale=scale})
      end
      do
        te_matrices[field] = stats.standardize(te_matrices[field],
                                               { center=center, scale=scale })
      end
      -- end
    end
  end
  
  print("# Data standarized")

  local costs_log1p = train:as_matrix("cost"):log1p()
  local train_costs,cost_center,cost_scale =
    stats.standardize( check( costs_log1p ), {
                         center = matrix(1,1,{stats.percentile(costs_log1p, 0.5)}), -- matrix(1,1):fill(0),
                         scale  = matrix(1,1):fill(1),
    })

  local tr_matrices_array,te_matrices_array = {},{}
  for _,fields in ipairs(result_order) do
    for _,field in ipairs(fields) do
      table.insert(tr_matrices_array, (assert(tr_matrices[field],
                                              "Not found train field "..field)))
      table.insert(te_matrices_array, (assert(te_matrices[field],
                                              "Not found test field "..field)))
    end
  end
  
  local train_data_joined = matrix.join(2, tr_matrices_array)
  local test_data_joined  = matrix.join(2, te_matrices_array)

  local train_tube_ids = train[{"tube_assembly_id"}]
  local test_tube_ids  = test[{"tube_assembly_id"}]
  local test_ids       = test[{"id"}]

  print("# Data preprocessed")
  
  return { train = train_data_joined,
           train_label = train_costs,
           test = test_data_joined,
           test_ids = test_ids,
           train_tube_ids = train_tube_ids,
           label_center = cost_center,
           label_scale = cost_scale,
           train_n_log_list = train_n_log_list,
           train_s_list = train_s_list,
           train_d_list = train_d_list,
           train_s_cats = train_s_cats,
           train_d_cats = train_d_cats,
           result_order = result_order,
  }
end

return {
  load_data = load_data,
  load_data_ram = load_data_ram,
  load_data_team = load_data_team,
}
