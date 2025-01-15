# Python习题

1.**序列解包**

~~~properties
给你两个字符串 s 和 t，每个字符串中的字符都不重复，且 t 是 s 的一个排列。

排列差 定义为 s 和 t 中每个字符在两个字符串中位置的绝对差值之和。

返回 s 和 t 之间的 排列差 。

 

示例 1：

输入：s = "abc", t = "bac"

输出：2

解释：

对于 s = "abc" 和 t = "bac"，排列差是：

"a" 在 s 中的位置与在 t 中的位置之差的绝对值。
"b" 在 s 中的位置与在 t 中的位置之差的绝对值。
"c" 在 s 中的位置与在 t 中的位置之差的绝对值。
即，s 和 t 的排列差等于 |0 - 1| + |1 - 0| + |2 - 2| = 2。

示例 2：

输入：s = "abcde", t = "edbac"

输出：12

解释： s 和 t 的排列差等于 |0 - 3| + |1 - 2| + |2 - 4| + |3 - 1| + |4 - 0| = 12。
~~~

~~~python
class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # 创建一个字典，将字符串 s 中的字符映射到它们的索引位置
        char2index = {c: i for i, c in enumerate(s)}

        # 计算字符串 t 中每个字符与其在字符串 s 中对应字符的索引差的绝对值之和
        return sum(abs(i - char2index[c]) for i, c in enumerate(t) if c in char2index)
~~~

2.**清除数字**

~~~properties
给你一个字符串 s 。

你的任务是重复以下操作删除 所有 数字字符：

删除 第一个数字字符 以及它左边 最近 的 非数字 字符。
请你返回删除所有数字字符以后剩下的字符串。

 

示例 1：

输入：s = "abc"

输出："abc"

解释：

字符串中没有数字。

示例 2：

输入：s = "cb34"

输出：""

解释：

一开始，我们对 s[2] 执行操作，s 变为 "c4" 。

然后对 s[1] 执行操作，s 变为 "" 。
~~~

~~~python
class Solution:
    def clearDigits(self, s: str) -> str:
        stack=[]
        for c in s:
            if c in "0123456789":
                stack.pop()
            else:
                stack.append(c)
        return "".join(stack)
~~~

3.排序

~~~properties
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

 

示例 1：

输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]
示例 2：

输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]
~~~

~~~python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return sorted([x**2 for x in nums])
~~~

~~~python
#sort和sorted的区别：
  
#sort 是列表对象的一个方法。
#它会对列表进行原地排序（即修改原列表），不会返回新的列表。
#只能用于列表。
nums = [3, 1, 2]
nums.sort()
print(nums)  # 输出: [1, 2, 3]
  
#sorted 是一个内置函数。
#它会返回一个新的已排序列表，不会修改原列表。
#可以用于任何可迭代对象（如列表、元组、字符串等）。  
nums = [3, 1, 2]
sorted_nums = sorted(nums)
print(sorted_nums)  # 输出: [1, 2, 3]
print(nums)  # 输出: [3, 1, 2]（原列表未修改）
~~~

