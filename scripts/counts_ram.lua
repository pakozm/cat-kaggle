local train = data_frame.from_csv("../input/ram/Train_features_final.csv", { NA="not applicable" })
for _,field in ipairs(train:get_columns()) do
  local m,dict = train:as_matrix(field,{dtype="categorical"})
  local counts = stats.ihist(m):select(2,3)
  local df = data_frame{ data={names=dict[1],counts=counts} }
  df:to_csv(field .. ".csv")
end
