# Python

# 1.基础

## 1.1模块化

向上/向下取整

~~~python
import math
math.floor(32.9)#32
math.ceil(32.3)#33
~~~

模块化

~~~python
from math import sqrt
sqrt(9)
~~~

## 1.2复数

计算负数平方根

~~~python
import cmath
cmath.sqrt(-1)#1j
~~~

## 1.3长字符串

Python可以识别单引号与双引号，为了防止混淆，长字符串使用三引号

~~~python
print('''This is a very long string.
It continues here."Hello World"
Still here.''')
~~~

## 1.4原始字符串

为防止大量使用转义字符

~~~python
print(r'C:\nowhere')#C:\nowhere
#一般将\单独作为一个字符
print(r'C:\Program Files''\\')#C:\Program Files\
~~~

# 2.容器

## 2.1序列

最常用序列包括列表和元组，**列表可修改**，**元组不可修改**。

### 2.1.1索引

负数索引，python将从最后一个元素开始往左数

~~~python
greeting ='Hello'
greeting[0]#'H'
greeting[-1]#o
~~~

如果函数调用返回一个序列，而你只需要一次，那么可以直接索引。

例如你只想获得用户输入年份的第四位，你可以这样做

~~~python
fourth=input('year:')[3]
#输入2005
print(fourth)#5
~~~



## 2.2映射

## 2.3集合（set）



