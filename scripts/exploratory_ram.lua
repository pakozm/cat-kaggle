--[[

Taken from:
https://www.kaggle.com/c/caterpillar-tube-pricing/forums/t/15001/ta-04114/83230

TA-00152 - 19
TA-00154 - 75
TA-00156 - 24
TA-01098 - 10
TA-01631 - 48
TA-03520 - 46
TA-04114 - 135
TA-17390 - 40
TA-18227 - 74
TA-18229 - 51

---------------------------------------------------------------------------

Taken from:
https://www.kaggle.com/c/caterpillar-tube-pricing/forums/t/15269/comp-threaded-csv-strange-value/85595

- I noticed that in the 193rd line of the file comp_threaded there is the value
  "See drawing". I guess it somehow sneaked in the data.

- Yes, that should have been removed. Should be 9999 meaning unknown.

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
    if not inv_filtered[field[i]] then field[i] = "9999" end
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
  local year,month,day = str:match("^(.+)%-(.+)%-(.+)$")
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
local train = data_frame.from_csv("../input/ram/Train_features_final.csv", { NA="NONE" })
-- local test  = data_frame.from_csv("../input/ram/Test_features_final.csv", { NA="NONE" })

-- with 9999 and numerical data: bend_radius (it counts 12 times)
-- with 9999 and categorical data: end_x

--[[
  "baffle" has 1 category. It can be ignored.
  "blind_hole" has 1 category.
  "flange" has only 1 category.
  "joint" has 1 category.
  "nozzle" has only 1 category.
  "requirements" has 1 category.
  "screw" has 1 category.
  "sheet" has 1 category.
  "spacer" has 1 category.
  "studded" has 1 category.
  
  "adapter" has 3 categories: 0, 1, 2. It will be better to merge 1 and 2 because category 2 is a singleton.
  "angle" has 2 categories: 0, 1. It can be ignored because category 1 is a singleton.
  "bar" has 2 categories: 0, 1. Category 1 has only 5 samples.
  "bellow" has 2 categories: 0, 1. Category 1 is a singleton.
  "block" has 3 categories: 0, 1, 2. However, category 2 counts twice.
  "body" has 2 categories: 0, 1. Category 1 counts 6.
  "bolt_nut" has 3 categories.
  "brazed" has 3 categories: 0, 1, 2. Category 1 counts 41 and 2 only 4 times.
  "bushing" has 2 categories and 1 counts 4 times only.
  "cap" has 3 categories with counts 20 and 4 for categories 1 and 2.
  "cast" with 2 categories, it has a singleton in category 1.
  "clamp" is like cast with 3 counts in category 1.
  "cleat" is like cast with 6 counts in category 1.
  "clip" is like cast with 33 counts in category 1.
  "collar" is like cast with 65 counts in category 1.
  "connector" has 6 categories, with counts 1 and 3 in categories 4 and 5.
  "count_of_components" has 14 categories.
  "count_of_specs" has 10 categories.
  "coupled" has 3 categories, with a singleton in category 2.
  "cover" has 2 categories, with 7 counts in category 1.
  "crude_oil" has 4 categories... is it quantitative or qualitative?
  "days_elapsed" has positive and negative values with very large dynamic range.
  "disc" has 2 categories by category 1 is a singleton.
  "distinct_components" has 9 categories.
  "elbow" has 3 categories.
  "end_a_1x" has 2 categories.
  "end_a_2x" has 2 categories.
  "end_a" is categorical.
  "end_category" is unknown for me.
  "end_check" is binary.
  "end_x_1x" is binary.
  "end_x_2x" is binary.
  "end_x" is categorical.
  "filler" has 3 categories, 2 and 4 counts in 1 and 2 categories.
  "fitted" has 4 categories, category 3 is a singleton.
  "groove" has 3 categories.
  "hanger_coat" has 4 categories.
  "hasp" is binary with 6 counts in category 1.
  "head" has 3 categories but category 2 counts 3 times.
  "hook" is binary.
  "hose" is binary but category 1 is a singleton.
  "housing" is binary.
  "hydraulic" has 3 categories but category 2 counts only 2 times.
  "inflation" has 4 possible values but it is numerical.
  "inner_volume" is numerical with a huge dynamical range.
  "length" is numerical with large dynamical range.
  "link" is binary.
  "lug" is binary".
  "manifold" is binary.
  "material_id" is categorical.
  "material_volume" is numerical with a huge dynamical range.
  "min_order_quantity" is numerical.
  "nipple" is binary.
  "num_bbbo" has 18 categories with singletons in 4 of them.
  "num_bends" has 18 categories with singletons in 4 of them.
  "num_boss" has 6 categories.
  "num_bracket" has 4 categories.
  "num_dim" is numerical.
  "num_end_for" has 3 categories.
  "orientation" has 5 categories.
  "orifice" is binary.
  "other_comp" has 9 categories.
  "other" has 9 categories.
  "outer_volume" is numerical with huge dynamical range.
  "pin" is binary.
  "pipe" has 5 categories.
  "plate" has 5 categories.
  "plating" is binary.
  "plug" has 4 categories.
  "post" is binary.
  "reducer" has 3 categories.
  "retainer" has 2 categories, but 2 times category 1.
  "ring" has 3 categories, but 3 times category 2.
  "riveted" is binary.
  "rod" is binary with singleton in category 1.
  "saddle" has 3 categories with singleton in category 2.
  "screen" has 3 categories but 2 and 1 times categories 1 and 2.
  "sdr" is numerical.
  "seal" has 3 categories.
  "shell" is binary.
  "shroud" is binary with singleton in category 1.
  "sleeve" has 3 categories.
  "socket" has 3 categories.
  "stem" is binary.
  "strap" is binary with singleton in category 1.
  "surface_area" is numerical with huge dynamic range.
  "tee" is binary.
  "tube" is binary.
  "union" has 3 categories (0, 1 and 4) but 4 is singleton.
  "unique" has 4 categories.
  "valve" has 3 categories (0, 1, and 3).
  "wall" is numerical.
  "washer" has 3 categories.
  "welded" has 5 categories.
  --
  "component8" is only used 3 times in training.
  "spec10" is NEVER used in training.
]]

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
}
local categorical_data_fields = {
  "supplier",
  "material_id",
  "end_a","end_x",
  "elbow",
  "hanger_coat",
  "plug","reducer",
  "seal","sleeve","socket",
  "other_comp","bolt_nut",
  "fitted","hydraulic",
  "orientation","unique","groove",
  "bin_count_specs","bin_moq",
}
local shared_categorical_data_fields = {
  "component_id_1","component_id_2","component_id_3","component_id_4",
  "component_id_5","component_id_6","component_id_7","component_id_8",
  "spec1","spec2","spec3","spec4","spec5","spec6","spec7","spec8","spec9","spec10",
}
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
}

remove_rare_events(train, nil, {"component_id_1",
                                "component_id_2",
                                "component_id_3",
                                "component_id_4",
                                "component_id_5",
                                "component_id_6",
                                "component_id_7",
                                "component_id_8",}, K)

remove_rare_events(train, nil, {"spec1",
                                "spec2",
                                "spec3",
                                "spec4",
                                "spec5",
                                "spec6",
                                "spec7",
                                "spec8",
                                "spec9",
                                "spec10",}, K)

for _,field in ipairs(categorical_data_fields) do
  remove_rare_events(train, nil, field, K)
end

split_date(train, "quote_date")

-----------------------------------------------------------------------------

-- numerical data
for field in iterator(numerical_asymmetric_data_fields) do
  local m = train:as_matrix(field)
  if field == "bend_radius" then m = m:index(1, m:lt(9999)) end
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
end

for field in iterator(numerical_log_gaussian_data_fields) do
  local m = train:as_matrix(field)
  if field == "bend_radius" then m = m:index(1, m:lt(9999)) end
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

for field in iterator(numerical_large_data_fields) do
  local m = train:as_matrix(field)
  if field == "bend_radius" then m = m:index(1, m:lt(9999)) end
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'sqrt-%s'"%{ field },
          stats.hist(matrix.op.sqrt(m), {breaks=20}))
end

local integer_data_fields = table.join(integer_data_fields,
                                       { "year", "month", "day", "dow" })
for field in iterator(integer_data_fields) do
  local m   = train:as_matrix(field)
  local idx = stats.levels(m)
  local df  = data_frame{ data={ id=idx, h=stats.ihist(m):select(2,4) },
                          columns={ "id", "h" } }
  gp.plot("'#1' u 1:2 w boxes title '%s'"%{ field }, df)
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

for field in iterator(numerical_other_data_fields) do
  local m = train:as_matrix(field)
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(matrix.op.clamp(m, 0.0, math.huge)), {breaks=20}))
end

-- numerical data
for field in iterator{ "cost" } do
  local m = train:as_matrix(field)
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

gp.set"xtics rotate by -45"
local data_fields = set(categorical_data_fields):update(set(shared_categorical_data_fields),set(binary_data_fields)):keys()
table.sort(data_fields)
for field in iterator(data_fields) do
  local m,idx = train:as_matrix(field, { dtype="categorical" })
  local df    = data_frame{ data={ id=idx[1], h=stats.ihist(m):select(2,4) },
                            columns={ "id", "h" } }
  gp.plot("'#1' u 0:2:xtic(1) w boxes title '%s'"%{ field }, df)
end
