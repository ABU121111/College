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

### 1.3连接数据库

~~~java
Class clazz = Class.forName("com.mysql.jdbc.Driver");
String url = "jdbc:mysql://localhost:3306/mysqlforjdbctest";
String user = "root";
String password = "111";
Connection connection = DriverManager.getConnection(url, user, password);
~~~

### 1.4 ResultSet

每次读取一整行，使用statement的excuteQuery来执行sql语句

~~~java
package com.hxh.jdbc;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.*;
import java.util.Date;
import java.util.Properties;

/**
 * @author 小黄小黄不再迷茫
 * @version 1.0
 */
public class ResultSetTest {
    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        Properties properties = new Properties();
        properties.load(new FileInputStream("src\\mysql.properties"));
        // 获取相关的值
        String user = properties.getProperty("user");
        String password = properties.getProperty("password");
        String url = properties.getProperty("url");
        String driver = properties.getProperty("driver");

        // 1. 注册驱动
        Class.forName(driver);

        // 2. 得到连接
        Connection connection = DriverManager.getConnection(url, user, password);

        // 3. 得到Statement
        Statement statement = connection.createStatement();

        // 4. 组织Sql
        String sql = "SELECT id, name, sex, birthday FROM student";
        // 执行给定的SQL语句，该语句返回单个 ResultSet 对象
        ResultSet resultSet = statement.executeQuery(sql);

        // 5. 使用while取出数据
        while (resultSet.next()){  // 让光标向后移动，如果没有更多行，则退出循环
            int id = resultSet.getInt(1);  // 获取该行的第一列
            String name = resultSet.getString(2);  // 获取该行第二列
            String sex = resultSet.getString(3);  // 获取该行第三列
            Date date = resultSet.getDate(4);  // 获取该行第四列
            System.out.println(id + "\t" + name + "\t" + sex + "\t" + date);
        }

        // 6. 关闭连接
        resultSet.close();
        statement.close();
        connection.close();
    }
}
~~~

### 1.5PreparedStatement

使用预处理操作防止恶意sql注入，并且还能使sql更简洁，无需+来连接

~~~java
package com.hxh.jdbc;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.*;
import java.util.Date;
import java.util.Properties;
import java.util.Scanner;

/**
 * @author 小黄小黄不再迷茫
 * @version 1.0
 */
public class LoginTest {
    public static void main(String[] args) throws IOException, SQLException, ClassNotFoundException {
        Scanner scanner = new Scanner(System.in);

        // 用户输入用户名和密码
        System.out.print("用户名：");
        String admin_name = scanner.nextLine();
        System.out.print("密码：");
        String admin_pwd = scanner.nextLine();

        // 通过Properties对象获取配置文件信息
        Properties properties = new Properties();
        properties.load(new FileInputStream("src\\mysql.properties"));
        // 获取相关的值
        String user = properties.getProperty("user");
        String password = properties.getProperty("password");
        String url = properties.getProperty("url");
        String driver = properties.getProperty("driver");

        // 1. 注册驱动
        Class.forName(driver);

        // 2. 得到连接
        Connection connection = DriverManager.getConnection(url, user, password);

        // 3. 得到 PreparedStatement
        // 3.1 组织Sql, ? 相当于占位符
        String sql = "SELECT name, pwd FROM admin WHERE name = ? AND pwd = ?";
        // 3.2 preparedStatement 对象实现了 PreparedStatement接口
        PreparedStatement preparedStatement = connection.prepareStatement(sql);
        // 3.3 给 ? 赋值
        preparedStatement.setString(1, admin_name);
        preparedStatement.setString(2, admin_pwd);

        // 4. 执行 select 使用 executeQuery, 如果执行的是 dml语句, 则使用 executeUpdate
        ResultSet resultSet = preparedStatement.executeQuery();
        if(resultSet.next()){// 如果查询到一条记录，则说明用户存在
            System.out.println("登录成功！");
        }else {
            System.out.println("登录失败！");
        }

        // 5. 关闭连接
        resultSet.close();
        preparedStatement.close();
        connection.close();
    }
}
~~~

## 2.Lombok

使用注解来简化代码，不必每次造轮子

我们通过添加@Getter和@Setter来为当前类的所有字段生成get/set方法，他们可以添加到类或是字段上，注意静态字段不会生成，final字段无法生成set方法。

我们还可以使用@Accessors来控制生成Getter和Setter的样式。

我们通过添加@ToString来为当前类生成预设的toString方法。

我们可以通过添加@EqualsAndHashCode来快速生成比较和哈希值方法。

我们可以通过添加@AllArgsConstructor和@NoArgsConstructor来快速生成全参构造和无参构造。

我们可以添加@RequiredArgsConstructor来快速生成参数只包含final或被标记为@NonNull的成员字段。

使用@Data能代表@Setter、@Getter、@RequiredArgsConstructor、@ToString、@EqualsAndHashCode全部注解。
一旦使用@Data就不建议此类有继承关系，因为equal方法可能不符合预期结果（尤其是仅比较子类属性）。

使用@Value与@Data类似，但是并不会生成setter并且成员属性都是final的。

使用@SneakyThrows来自动生成try-catch代码块。

使用@Cleanup作用与局部变量，在最后自动调用其close()方法（可以自由更换）

使用@Builder来快速生成建造者模式。
通过使用@Builder.Default来指定默认值。
通过使用@Builder.ObtainVia来指定默认值的获取方式。
