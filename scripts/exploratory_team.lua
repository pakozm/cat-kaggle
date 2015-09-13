--[[

  train = train.drop(['supplier','material_id' , 'quote_date'
  ,'tube_assembly_id', 'cost', 'component_id_4', 'component_id_5',
  'component_id_6',
  'component_id_7','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8',
  'component_id_8' ], axis = 1)
  
]]

local DATE_DELTA = 1
local K = 10
local gp = require "april_tools.gnuplot"

gp.verbosity(1)
gp.set"terminal pdf"
gp.set"output 'exploratory.pdf'"
gp.set"style fill solid border -1"

local function filter(df, field, filtered)
  local field = df[{field}]
  local inv_filtered = table.invert(filtered)
  for i=1,#field do
    if not inv_filtered[field[i]] then field[i] = 9999 end
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
  return filtered
end

local function remove_rare_events(train, test, fields, k)
  if type(fields) ~= "table" then fields = {fields} end
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
  print(fields[1], #dict, "=>", #filtered)
  for i,field in ipairs(fields) do
    filter(train, field, filtered)
    if test then filter(test, field, filtered) end
  end
end

-- given a date string returns a table with date values
local function date_parser(str)
  local year,month,day = str:match("^(.+)%-(.+)%-(.+) .*$")
  return {
    year  = tonumber(year),
    month = tonumber(month),
    day   = tonumber(day),
  }
end

-- Receives a data_frame and the date field and splits it into year, month, day
-- and dow (day-of-week). At the end, it DROPS the date field.
local function split_date(df, field)
  -- auxiliary columns
  df[{"timestamp"}] = df:parse_datetime(field,  date_parser)
  df[{"datetime"}]  = df:map("timestamp", bind(os.date, "*t"))
  -- date columns
  df[{"year"}]  = df:map("datetime", function(x) return math.floor(x.year/DATE_DELTA)*DATE_DELTA end)
  df[{"month"}] = df:map("datetime", function(x) return x.month end)
  df[{"day"}]   = df:map("datetime", function(x) return x.day end)
  df[{"dow"}]   = df:map("datetime", function(x) return x.wday end)
  -- drop all auxiliary columns
  df:drop(2, "timestamp")
  df:drop(2, "datetime")
  -- drop date field
  df:drop(2, field)
end

-- ,tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,
-- bracket_pricing,quantity,cost,material_id,diameter,wall,length,num_bends,
-- bend_radius,end_a_1x,end_a_2x,end_x_1x,end_x_2x,end_a,end_x,num_boss,
-- num_bracket,other,s_mean,s_median,s_sd,m_mean,m_median,m_sd,component_id_1,
-- quantity_1,component_id_2,quantity_2,component_id_3,quantity_3,component_id_4,
-- quantity_4,component_id_5,quantity_5,component_id_6,quantity_6,component_id_7,
-- quantity_7,component_id_8,quantity_8,weight,num_unique_features,
-- num_orientations,connection_type_counts,interactions,num_comp,num_spec,
-- SP.0063,SP.0012,SP.0080,SP.0007,SP.0026,SP.0082,SP.0069,SP.0070

local train = data_frame.from_csv("../input/team/TRAIN.csv", { NA="NONE" })

-- with 9999 and numerical data: bend_radius (it counts 12 times)
-- with 9999 and categorical data: end_x

--[[
  
]]

local numerical_data_fields = {
  "annual_usage",
  "min_order_quantity",
  "quantity",
  "cost",
  "diameter",
  "wall",
  "length",
  "num_bends",
  "bend_radius",
  "num_boss",
  "num_bracket",
  "other",
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
  "weight",
  "num_unique_features",
  "num_orientations",
  "connection_type_counts",
  "interactions",
  "num_comp",
  "num_spec",
}

local categorical_data_fields = {
  "supplier",
  "bracket_pricing",
  "material_id",
  "end_a_1x",
  "end_a_2x",
  "end_x_2x",
  "end_a",
  "end_x",
  "component_id_1",
  "component_id_2",
  "component_id_3",
  "component_id_4",
  "component_id_5",
  "component_id_6",
  "component_id_7",
  "component_id_8",
  "SP.0063",
  "SP.0012",
  "SP.0080",
  "SP.0007",
  "SP.0026",
  "SP.0082",
  "SP.0069",
  "SP.0070",
}

remove_rare_events(train, nil, {"component_id_1",
                                "component_id_2",
                                "component_id_3",
                                "component_id_4",
                                "component_id_5",
                                "component_id_6",
                                "component_id_7",
                                "component_id_8",}, K)

-- for _,field in ipairs(categorical_data_fields) do
--   remove_rare_events(train, nil, field, K)
-- end

split_date(train, "quote_date")

-----------------------------------------------------------------------------

-- numerical data
for field in iterator(numerical_data_fields) do
  local m = train:as_matrix(field)
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'sqrt-%s'"%{ field },
          stats.hist(matrix.op.sqrt(m), {breaks=20}))
end

local integer_data_fields = table.join({ "year", "month", "day", "dow" })
for field in iterator(integer_data_fields) do
  local m   = train:as_matrix(field)
  local idx = stats.levels(m)
  local df  = data_frame{ data={ id=idx, h=stats.ihist(m):select(2,4) },
                          columns={ "id", "h" } }
  gp.plot("'#1' u 1:2 w boxes title '%s'"%{ field }, df)
end
