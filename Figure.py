

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
# x = [np.random.randint(0, 1) for i in range(100)]
# print(x)
# y = np.random.rand(100)
# print(y)

# arr = np.zeros((100, 2))  # arr 热力图中的值阵
#
# for i in range(len(x)):
#     arr[y[i], x[i]] = y[i]
# z = (np.random.rand(6)+np.linspace(0,1, 6)).reshape(3, 2)
s = '/Users/shengbinjia/Documents/GitHub/TripleConfidence/data/result/'
f0 = open(s + 'Model1_model_TransE_---train_conf0.txt', 'r')
f1 = open(s + 'Model1_model_TransE_---train_conf1.txt', 'r')
list0 = []
lines0 = f0.readlines()
num = 0
for i, l0 in enumerate(lines0):
        list0.append([float(l0)])
        if float(l0)>0.5:
                if i <100:
                        num +=1
f0.close()
print(num / 10)

list1 = []
lines1 = f1.readlines()
for l1 in lines1:
        list1.append([float(l1)])
f1.close()


# z= np.concatenate((list0, list1), axis=1)
# print(z)
# plt.imshow(z, extent=(0, .1, 0, 1),
#         cmap=cm.hot, norm=LogNorm())
# plt.colorbar()
# plt.show()

# N = 50 # 点的个数
# x = np.random.rand(N) * 2 # 随机产生50个0~2之间的x坐标
# y = np.random.rand(N) * 2 # 随机产生50个0~2之间的y坐标
# colors = np.random.rand(N) # 随机产生50个0~1之间的颜色值
# area = np.pi * (15 * np.random.rand(N))**2  # 点的半径范围:0~15

n0 = np.random.rand(75000) * 0.49
x = np.concatenate((np.zeros(75000) + n0, np.ones(75000) - n0), axis=0)
print(x)
y= np.concatenate((list0[:75000], list1[:75000]), axis=0)
print(y.shape)

# 画散点图
plt.ylabel("Trustworthiness")
colors = x # 随机产生50个0~1之间的颜色值
plt.scatter(x, y,c = 'b', s=0.05)
plt.savefig(s + 'sandian.pdf')#先存，再show
plt.show()



