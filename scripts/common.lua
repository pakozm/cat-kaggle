local function shuffle(rng, a, b, tube_ids)
  assert(a:dim(1) == b:dim(1))
  local perm
  if not tube_ids then
    error("BIASED")
    perm = matrixInt32(rng:shuffle(a:dim(1)))
  else
    perm = matrixInt32(a:dim(1))
    local unique_tube_ids = iterator(pairs(iterator(tube_ids):map(lambda'|x|x,true'):table())):select(1):table()
    table.sort(unique_tube_ids)
    local tube_inv_perm = table.invert( rng:shuffle(unique_tube_ids) )
    local perms_table = iterator.range(#unique_tube_ids):map(lambda'|i|i,{}'):table()
    for i,tube_id in ipairs(tube_ids) do
      local p = tube_inv_perm[tube_id]
      perms_table[p] = perms_table[p] or {}
      table.insert(perms_table[p], i)
    end
    local k=0
    for _,tbl in ipairs(perms_table) do
      for _,i in ipairs(tbl) do
        k=k+1
        perm[k] = i
      end
    end
    do
      local d={} for i,k in ipairs(perm) do assert(not d[k]) d[k] = true end
    end
  end
  a = a:index(1, perm)
  b = b:index(1, perm)
  return a,b,perm
end

return {
  shuffle = shuffle,
}
