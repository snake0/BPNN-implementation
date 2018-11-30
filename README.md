# BPNN-implementation

## 实现细节
环境：macOS 10.14.1, PyCharm Professional

这次作业实现了一个有 M 维输入层、 N 维输出层， K个维数不同的隐藏层的神经网络。（M、N、K均任意）

其激活函数有三种：

* Sigmoid 函数

![0](https://img-blog.csdn.net/20181007212304602?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNzMxODYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![](https://img-blog.csdn.net/20181007212442598?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNzMxODYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* tanh 函数

![0](https://img-blog.csdn.net/2018100721470943?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNzMxODYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![](https://img-blog.csdn.net/20181007214734148?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNzMxODYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* 线性函数 y = x

同时使用了 back-propagation 算法，更新每个节点的数据以及权重。
使用权重以及激活函数更新节点数据

```python
# update hidden layers

        for k in range(self.layers_num - 1):
            for j in range(self.hidden_layer_nums[k + 1]):
                act = 0.0
                for i in range(self.hidden_layer_nums[k]):
                    act += self.hidden_layer_actions[k][i] * self.hidden_weights[k][i][j]
                self.hidden_layer_actions[k + 1][j] = act_fun(act)
```
使用激活函数的导数以及误差更新权重
```python
# calculate output error for each hidden layer

        for i in range(self.layers_num - 1):
            for j in range(self.hidden_layer_nums[self.layers_num - i - 2]):
                error = 0.0
                for k in range(self.hidden_layer_nums[self.layers_num - i - 1]):
                    error += self.hidden_delta[self.layers_num - i - 1][k] \
                             * self.hidden_weights[self.layers_num - i - 2][j][k]
                self.hidden_delta[self.layers_num - i - 2][j] = \
                    d_act_fun(self.hidden_layer_actions[self.layers_num - i - 2][j]) * error
```

## 功能测试
* 基本测试
使用课件中的实验数据
```python
pat=[
[[0.05],[0.1]],
[[0.01], [0.99]]
]
```
经过一百次迭代得到结果：
[0.05, 0.1]  :  [0.01, 0.9899999999999922]
误差以足够小
* 扩展功能
  实现半加器，输入加数a(i),b(i)和进位数c(i-1)，可以输出和数s(i+1)和进位数c(i)

  测试数据：
```python
pat = [
[[0,0,0],[0,0]],
[[0,0,1],[1,0]],
[[0,1,0],[1,0]],
[[0,1,1],[0,1]],
[[1,0,0],[0,1]],
[[1,0,1],[0,1]],
[[1,1,0],[0,1]],
[[1,1,1],[1,1]],
]
```

  100次迭代后测试结果：
0,0,0  :  [-0.0007395855442126433, -0.002862822764023466]
0,0,1  :  [0.9992560979993417, 0.00283586860296436]
0,1,0  :  [0.9995193940093299, 0.0010901015201309635]
0,1,1  :  [2.3725283893951964e-05, 0.9992334935962858]
1,0,0  :  [0.0015614782081181185, 0.9996071759393457]
1,0,1  :  [0.002630613587701749, 0.999459717950869]
1,1,0  :  [0.0012913221829761185, 0.9996194056484254]
1,1,1  :  [0.9992946200138516, 0.9994577979646382]

误差足够小