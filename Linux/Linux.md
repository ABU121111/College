# Linux

## 1.安装

~~~bash
sudo apt install postgresql postgresql-contrib

//redis
sudo apt isntall redis-server
~~~

具体解释如下：

1. **sudo**: 以超级用户权限执行命令。
2. **apt**: 包管理工具，用于安装、更新和删除软件包。
3. **install**: 安装指定的软件包。
4. **postgresql**: PostgreSQL数据库的主包。
5. **postgresql-contrib**: 包含额外的功能和扩展的附加包。

## 2.检查版本

~~~bash
psql --version
psql (PostgreSQL) 16.6 (Ubuntu 16.6-0ubuntu0.24.04.1)
~~~

## 3.运行命令

~~~bash
sudo service postgresql start
sudo service postgresql status
sudo service postgresql stop
sudo service postgresql restart

//redis
sudo service redis-server start
//mysql
sudo mysql -u root
~~~

## 4.添加密码

~~~bash
sudo passwd postgres
New password:
Retype new password:

passwd: password updated successfully
~~~

## 5.运行

~~~bash
sudo -u postgres psql

redis-cli
~~~

退出使用快捷键Ctrl+D

使用\du命令显示账户

## 6.远程访问

~~~bash
cd /etc/postgresql/16/main/
/etc/postgresql/16/main$ ls -la
~~~

1. **`ls`**：列出目录内容的命令。
2. **`-l`**：以长格式（详细信息）列出文件和目录。
3. **`-a`**：显示所有文件，包括隐藏文件（以`.`开头的文件）。

## 7.新建

切换到默认用户

~~~bash
sudo -u postgres psql
~~~

创建新的用户和数据库

~~~sql
CREATE USER Bolt WITH PASSWORD '123123';
~~~

