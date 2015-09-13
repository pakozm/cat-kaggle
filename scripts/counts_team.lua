local train = data_frame.from_csv("../input/team/TRAIN.csv", { NA="NONE" })
for _,field in ipairs(train:get_columns()) do
  local m,dict = train:as_matrix(field,{dtype="categorical"})
  local counts = stats.ihist(m):select(2,3)
  local df = data_frame{ data={names=dict[1],counts=counts} }
  df:to_csv("TEAM/" .. field .. ".csv")
end
