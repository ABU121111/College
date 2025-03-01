# Myabtis

## 1.基础配置

使用xml写基础配置文件mybatis-config.xml，对数据库进行连接配置

~~~xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
  "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
  <environments default="development">
    <environment id="development">
      <transactionManager type="JDBC"/>
      <dataSource type="POOLED">
        <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://localhost:3306/student_score"/>
        <property name="username" value=""/>
        <property name="password" value=""/>
      </dataSource>
    </environment>
  </environments>
  <mappers>
    <mapper resource="TestMapper.xml"/>
</mappers>

</configuration>
~~~

然后对读取数据库的方式进行配置TsetMapper.xml，这样就不需要直接在代码中写sql语句

~~~xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="TestMapper">
    <select id="selectStudent" resultType="com.tsAdmin.entity.Student">
        select * from student_score
    </select>
    <select id="selectStudentByName" parameterType="String" resultType="com.tsAdmin.entity.Student">
    SELECT * FROM student_score WHERE name = #{name}
    </select>
</mapper>
~~~

随后可以用SqlSessionFactory，SqlSession，来select数据库中的数据

~~~java
public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(new FileInputStream("src/main/resources/mybatis-config.xml"));
        try (SqlSession sqlSession = sqlSessionFactory.openSession(true)) {
            List<Student> list = sqlSession.selectList("selectStudent");
            list.forEach(System.out::println);
            Student st=sqlSession.selectOne("selectStudentByName","t");
            System.out.println(st);
        }
    }
}
~~~



