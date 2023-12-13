import numpy as np
import drjit as dr

# 创建一个NumPy数组
np_array = np.array([1, 2, 3, 4])

# 转换为Dr.Jit数组
x = dr.ones(dr.llvm.ad.Array3f,shape=100)
print(dr.slice(x,10))
