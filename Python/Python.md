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
#还可以直接数乘
print(7*['th'])#['th', 'th', 'th', 'th', 'th', 'th', 'th']
~~~

### 2.1.2切片

切片访问指定范围内元素，左闭右开

~~~python
numbers=[1,2,3,4,5,6,7,8,9,10]
numbers[3:6]#[4,5,6]
numbers[0:1]#[1]
~~~

也可以用负数索引从后往前，但是无法包括最后一个元素

~~~python
numbers[-3:-1]#[8,9]
~~~

为了包括边界位置元素，可使用简写

~~~python
numbers[-3:]#[8,9,10]
numbers[:3]#[1,2,3]
~~~

还可以复制整个序列

~~~python
numbers[:]#[1,2,3,4,5,6,7,8,9,10]
~~~

还可以显式指定步长

~~~python
numbers[0:10:2]#[1,3,5,7,9]
#简写，每三个取一个步长为4
numbers[::4]#[1,5,9]
~~~

负数步长必须**第一个索引比第二个大**

~~~python
#负数步长从右向左,满足第一位在第二位之后
numbers[8:3:-2]#[10,8,6,4,2]
#负数步长默认从右到左
numbers[5::-2]#[6,4,2]
numbers[:5:-2]#[10,8]
~~~

### 2.1.3成员资格

Python使用in来判断变量是否在序列中

~~~python
print('P' in 'Python')#True
~~~

### 2.1.4列表-Python的主力

由于不能像修改列表那样修改字符串，所以使用list

~~~python
list('Hello')#['H','e','l','l','o']
~~~

#### 2.1.4.1基本列表操作

**修改元素**

~~~python
x=[1,1,1]
x[1]=2
#[1,2,1]
~~~

**删除元素**

~~~python
del x[2]
#[1,1]
~~~

**给切片赋值**

~~~python
name=list['Perl']
name[2:]=list('ar')
>>>['P','e','a','r']
~~~

Ps1：替换切片

~~~python
name=list('Perl')
name[1:]=list('ython')
>>>['P', 'y', 't', 'h', 'o', 'n']
~~~

Ps2：不改变原元素插入新元素

~~~python
numbers=[1,5]
numbers[1:1]=[2,3,4]
>>>[1, 2, 3, 4, 5]
~~~

Ps3：删除切片

~~~python
numbers=[1,5]
numbers[1:1]=[2,3,4]
del numbers[1:4]
>>>[1, 5]
~~~

#### 2.1.4.2列表方法

**1.append**

将一个**对象**添加到列表的末尾

~~~python
numbers=[1,5]
numbers.append(3)
>>>[1, 5, 3]
numbers.append(numbers)
>>>[1, 5, 3,[1, 5, 3]]
~~~

**2.clear**

清空列表内容，类似删除

~~~python
numbers.clear()
>>>[]
~~~

**3.copy**

复制列表，指向副本

~~~python
b=number.copy()
>>>[1,5,3]
~~~

**4.count**

计算指定元素在列表中出现了多少次

~~~python
numbers=[1,5,4,3,4,2,1]
print(numbers.count(4))
>>>2
~~~

**5.extend**

将多个值附加到列表末尾

~~~python
numbers=[1,5,3]
numbers.extend(numbers)
>>>[1, 5, 3, 1, 5, 3]
~~~

**6.index**

在列表中查找指定元素第一次出现的索引

~~~python
numbers=[1,5,4,3,4,2,1]
print(numbers.index(4))
>>>2
~~~

**7.insert**

将一个对象插入列表

~~~python
numbers=[1,5,4,3,4,2,1]
numbers.insert(3,'four')
>>>[1, 5, 4, 'four', 3, 4, 2, 1]
~~~

**8.pop**

从列表中删除元素，默认是最后一个,并且返回该元素

~~~python
numbers=[1,5,4,3,4,2,1]
x=numbers.pop(3)
>>>[1, 5, 4, 4, 2, 1]
>>>x=3
~~~

**9.remove**

删除第一个为指定值的元素，修改列表但是没有返回值

~~~python
numbers=[1,5,4,3,4,2,1]
numbers.remove(4)
>>>[1, 5, 3, 4, 2, 1]
~~~

**10.reverse**

反转列表

~~~python
numbers=[1,5,4,3,4,2,1]
numbers.reverse()
>>>[1, 2, 4, 3, 4, 5, 1]
~~~

**11.sort**

对源列表排序，修改列表且没有返回值，为此可以使用**sorted**函数

~~~python
numbers=[1,5,4,3,4,2,1]
temp=sorted(numbers)
print(numbers)
print(temp)
>>>[1, 5, 4, 3, 4, 2, 1]
>>>[1, 1, 2, 3, 4, 4, 5]
~~~

Ps：高级排序，sorted接受参数**key**和**reverse**，分别代表**排序函数**和**是否反转**

~~~python
temp=sorted(numbers,reverse=True,key=len)
~~~

### 2.1.5元组-不可修改的序列

元组不可修改，一般用（）和，表示

~~~python
x=3*(40+2)#126
x=3*(42+2,)#(42,42,42)
~~~

使用tuple将序列转换为元组

~~~python
tuple([1,2,3])
>>>(1,2,3)
~~~

## 2.2映射

## 2.3集合（set）

## 3.使用字符串

