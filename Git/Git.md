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
git remote add origin git@github.com:ABU121111/college.git
~~~

如果出现rejected，就是没有连接到分支

~~~properties
git pull --rebase orifin master
~~~

6.推送

开始为空时加“-u”，非空则不需要-u

~~~properties
git push -u origin master
~~~

## 1.1提交特定分支

1.初始化本地仓库

~~~bash
git init
~~~

2.远程连接

~~~bash
git remote add origin git@github.com:ABU121111/college.git
~~~

3.拉取分支

~~~bash
git fetch origin
~~~

4.选择分支

~~~bash
git checkout -b name origin/name
~~~

5.设置上游分支

~~~bash
git pull --set-upstream origin name
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

## 3.更新

1.拉取

~~~properties
git pull origin master
~~~

2.添加

~~~properties
git add .
~~~

3.提交

~~~properties
git commit -m "new"
~~~

4.推送

~~~properties
git push origin master
~~~

