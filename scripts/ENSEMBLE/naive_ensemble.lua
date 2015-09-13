--[[
  This file is part of ESAI-CEU-UCH/kaggle-epilepsy (https://github.com/ESAI-CEU-UCH/kaggle-epilepsy)
  
  Copyright (c) 2014, ESAI, Universidad CEU Cardenal Herrera,
  (F. Zamora-Martínez, F. Muñoz-Malmaraz, P. Botella-Rocamora, J. Pardo)
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
]]
local inv_columns = {}
local f = {}
for i=1,#arg do
  f[i] = assert( io.open(arg[i]), "Unable to open " .. arg[i] )
  inv_columns[i] = table.invert(f[i]:read("*l"):gsub('"',''):tokenize(","))
end
local K = #arg
print("id,cost")
local finished = false
while true do
  local result,id,c = 0
  local lines = iterator(ipairs(f)):select(2):call("read","*l"):enumerate()
  for i,line in lines do
    if not line then
      finished=true
    else
      local tbl,id2 = table.pack( line:match("(.+)%,(.+)") )
      id2,c = tbl[inv_columns[i].id], tbl[inv_columns[i].cost]
      assert(not id or id==id2)
      id = id or id2
      result = result + tonumber(c)
    end
  end
  if not id then break end
  printf("%s,%g\n", id, result/K)
end
