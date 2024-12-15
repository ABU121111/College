# Git

## 1.提交

1.初始化仓库

~~~properties
git init
~~~

2.添加

加入所有文件

~~~properties
git add .
~~~

加入特定文件

~~~properties
git add 文件名
~~~

3.查询状态

~~~properties
git status
~~~

4.提交

~~~properties
git commit -m"说明"
~~~

5.远程连接

~~~properties
git remote add origin git@github.com:ABU12111/college.git
~~~

6.推送

开始为空时加“-u”，非空则不需要-u

~~~properties
git push -u origin master
~~~

## 2.删除

1.预览

~~~properties
git rm -r -n --cached 文件夹或文件
~~~

2.删除

~~~properties
git rm -r --cached 文件或文件夹
~~~

3.提交

~~~properties
git commit -m"temp"
~~~

4.推送

~~~properties
git push
~~~

                                   