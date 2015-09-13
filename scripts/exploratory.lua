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
    if not inv_filtered[field[i]] then field[i] = nan end
  end
end

local function remove_rare_events(train, test, field, k)
  local m,dict = train:as_matrix(field, { dtype="categorical" })
  local dict   = dict[1]
  local counts = stats.ihist(m):select(2,3)
  local counts = counts:gt(k):to_index():flatten()
  local filtered = {}
  for i=1,#counts do filtered[i] = dict[counts[i]] end
  filter(train, field, filtered)
  if test then filter(test,  field, filtered) end
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

-- tube_assembly_id,supplier,quote_date,annual_usage,min_order_quantity,bracket_pricing,quantity,cost
local train = data_frame.from_csv("../input/train_set.csv", { NA="NONE" })
-- tube_assembly_id,material_id,diameter,wall,length,num_bends,bend_radius,end_a_1x,end_a_2x,end_x_1x,end_x_2x,end_a,end_x,num_boss,num_bracket,other
local tube = data_frame.from_csv("../input/tube.csv", { NA="NONE" })

local merged = train:merge(tube, { how="left", key="tube_assembly_id" })

remove_rare_events(merged, nil, "supplier", K)
remove_rare_events(merged, nil, "material_id", K)
remove_rare_events(merged, nil, "end_a", K)
remove_rare_events(merged, nil, "end_x", K)

split_date(merged, "quote_date")

-- numerical data
for field in iterator{ "diameter", "wall", "length", "bend_radius" } do
  local m = merged:as_matrix(field)
  if field == "bend_radius" then m = m:index(1, m:lt(9999)) end
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

for field in iterator { "num_bends", "num_boss", "num_bracket", "other",
                        "year", "month", "day", "dow" } do
  local m   = merged:as_matrix(field)
  local idx = stats.levels(m)
  local df  = data_frame{ data={ id=idx, h=stats.ihist(m):select(2,4) },
                          columns={ "id", "h" } }
  gp.plot("'#1' u 1:2 w boxes title '%s'"%{ field }, df)
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

-- numerical data
for field in iterator{ "annual_usage", "min_order_quantity", "quantity" } do
  local m   = merged:as_matrix(field)
  local idx = stats.levels(m)
  local df  = data_frame{ data={ id=idx, h=stats.ihist(m):select(2,4) },
                          columns={ "id", "h" } }
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field }, stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
  print(field, #merged:levels(field))
end

-- numerical data
for field in iterator{ "cost" } do
  local m = merged:as_matrix(field)
  gp.plot("'#1' u 2:4 w boxes title '%s'"%{ field },
          stats.hist(m, {breaks=20}))
  gp.plot("'#1' u 2:4 w boxes title 'log1p-%s'"%{ field },
          stats.hist(matrix.op.log1p(m), {breaks=20}))
end

gp.set"xtics rotate by -45"

-- "tube_assembly_id"
for field in iterator { "supplier", "material_id", "end_a", "end_x" } do
  local m,idx = merged:as_matrix(field, { dtype="categorical" })
  local df    = data_frame{ data={ id=idx[1], h=stats.ihist(m):select(2,4) },
                            columns={ "id", "h" } }
  gp.plot("'#1' u 0:2:xtic(1) w boxes title '%s'"%{ field }, df)
end
