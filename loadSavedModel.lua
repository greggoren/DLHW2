

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'math'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

print(trainData:size())


saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
print(classes[trainLabels[100]]) -- display the 100-th image class
do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/3)
      image = require 'image'
	  for i=1,input:size(1) do
		if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      

	end
	
    end
    self.output:set(input:cuda())
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end


local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


--  ****************************************************************
--  Define our neural network
--  ****************************************************************
function loadModel(filename)
	return torch.load(filename)
end
local model = nn.Sequential()
model = loadModel('modelFinal.txt')
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 32
local optimState = {}

function forwardNet(data,labels, train)
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
		local yt = labels:narrow(1, i, batchSize)
		local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
   
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

---------------------------------------------------------------------

epochs = 1
testLoss = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
--model:apply(function(l) l:reset() end)

timer = torch.Timer()
print("good...very good")
for e = 1, epochs do
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5== 0 then
        print('Epoch ' .. e .. ':')
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end


--  ****************************************************************
--  Network predictions
--  ****************************************************************
--[[
model:evaluate()   --turn off dropout
]]
print(classes[testLabels[10]])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end
