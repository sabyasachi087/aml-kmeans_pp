from scipy import spatial

dataSetI = [4, 3]
dataSetII = [3, 4]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
print(result)
