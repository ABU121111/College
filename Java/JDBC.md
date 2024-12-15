# JDBC

## 1.基础

### 1.1基础开发样例

~~~java
public static void main(String[] args) throws SQLException {
        //建立连接
        Connection conn= DriverManager.getConnection("jdbc:mysql://localhost:3306/test","root","");//jdbc协议，mysql子协议，localhost:3306/test子名称，不同的数据库厂商有不同的协议

        //创建声明
        Statement statement=conn.createStatement();

        //执行sql
        statement.execute("insert into tb values(1,'电子科大','成都') ");

        //关闭资源
        statement.close();
        conn.close();
        }
~~~

### 1.2 Statement用法

~~~properties
Boolean execute(String sql):执行sql语句，如果有结果返回true，否则返回false
int executeUpdate(String sql):执行sql语句，返回一个int值，表示影响了几条记录
例如修改记录，返回修改的记录数量：
~~~



