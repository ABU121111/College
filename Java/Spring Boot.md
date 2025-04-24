# Spring Boot

## 1.快速开发

### 1.1常用模块整合

一开始未导入其他依赖，只导入最基本的依赖：

~~~xml
<dependency>
     <groupId>org.springframework.boot</groupId>
     <artifactId>spring-boot-starter</artifactId>
</dependency>
~~~

SpringMVC相关依赖，内置tomcat服务器

~~~xml
<dependency>
     <groupId>org.springframework.boot</groupId>
     <artifactId>spring-boot-starter-web</artifactId>
</dependency>
~~~

创建一个Controller，便可直接访问

~~~java
@Controller
public class TestController {
    
    @ResponseBody
    @GetMapping("/")
    public String index(){
        return "Hello World";
    }
}
~~~

同时也可以直接返回JSON数据给前端

~~~java
@Data
public class Student {
    int sid;
    String name;
    String sex;
}
~~~

~~~java
@ResponseBody
@GetMapping("/")
public Student index(){
    Student student = new Student();
    student.setName("小明");
    student.setSex("男");
    student.setSid(10);
    return student;
}
~~~

同样的，我们也可以整合Thymeleaf框架

~~~xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-thymeleaf</artifactId>
</dependency>
~~~

在默认情况下，我们需要在`resources`目录下创建两个目录：

![image-20230715225833930](https://oss.itbaima.cn/internal/markdown/2023/07/15/HfGt61A7OqVDesz.png)

这两个目录是默认配置下需要的，名字必须是这个：

- `templates` - 所有模版文件都存放在这里
- `static` - 所有静态资源都存放在这里

在controller中不用写任何内容，默认将index.html作为首页文件

~~~java
@Controller
public class TestController {
		//什么都不用写
}
~~~

MyBatis也是一样，导入依赖和MySQL驱动

~~~xml
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>3.0.3</version>
</dependency>
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-j</artifactId>
    <scope>runtime</scope>
</dependency>
~~~

### 1.2自定义运行器

在项目中，可能会遇到这样一个问题：我们需要在项目启动完成之后，紧接着执行一段代码。

我们可以编写自定义的ApplicationRunner来解决，它会在项目启动完成后执行：             

```java
@Component
public class TestRunner implements ApplicationRunner {
    @Override
    public void run(ApplicationArguments args) throws Exception {
        System.out.println("我是自定义执行！");
    }
}
```

当然也可以使用CommandLineRunner，它也支持使用@Order或是实现Ordered接口来支持优先级执行。

### 1.3配置文件

首先配置数据库以及端口等底层信息，在application.yml中

~~~yaml
server:
  port: 8080

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/student_score
    username: 
    password:
    driver-class-name: com.mysql.cj.jdbc.Driver
~~~

然后创建实体类

~~~java
@Data
@AllArgsConstructor
public class Student {
    int id;
    String name;
    int score_one;
    int score_two;
}
~~~

接着映射，直接打@Mapper注解即可自动配置注册

~~~java
@Mapper
public interface StudentMapper {
    @Select("select * from student_score where id=#{id}")
    Student findStudentById(int id);
}

~~~

最后在TestController中指定路径

~~~java
@Controller
public class TestController {

    @Autowired
    StudentMapper studentMapper;


    @GetMapping("/test")
    @ResponseBody
    public Student index(){
        return studentMapper.findStudentById(2023090912);
    }
}
~~~

## 2.日志系统

### 2.1打印日志信息

~~~java
//注解
@Slf4j
@Controller
public class TestController {
    @PostConstruct
    public void init(){
        log.info("我是你的谁");

    }
}
~~~

### 2.2配置Logback

只需新建logback-spring.xml即可

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <!--  导入其他配置文件，作为预设  -->
    <include resource="org/springframework/boot/logging/logback/defaults.xml" />

    <!--  Appender作为日志打印器配置，这里命名随意  -->
    <!--  ch.qos.logback.core.ConsoleAppender是专用于控制台的Appender  -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>${CONSOLE_LOG_PATTERN}</pattern>
            <charset>${CONSOLE_LOG_CHARSET}</charset>
        </encoder>
    </appender>

    <!--  指定日志输出级别，以及启用的Appender，这里就使用了我们上面的ConsoleAppender  -->
    <root level="INFO">
        <appender-ref ref="CONSOLE"/>
    </root>
</configuration>
~~~

接着我们来看看如何开启文件打印，我们只需要配置一个对应的Appender即可：                        

```xml
<!--  ch.qos.logback.core.rolling.RollingFileAppender用于文件日志记录，它支持滚动  -->
<appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
    <encoder>
        <pattern>${FILE_LOG_PATTERN}</pattern>
        <charset>${FILE_LOG_CHARSET}</charset>
    </encoder>
    <!--  自定义滚动策略，防止日志文件无限变大，也就是日志文件写到什么时候为止，重新创建一个新的日志文件开始写  -->
    <rollingPolicy class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
        <!--  文件保存位置以及文件命名规则，这里用到了%d{yyyy-MM-dd}表示当前日期，%i表示这一天的第N个日志  -->
        <FileNamePattern>log/%d{yyyy-MM-dd}-spring-%i.log</FileNamePattern>
        <!--  到期自动清理日志文件  -->
        <cleanHistoryOnStart>true</cleanHistoryOnStart>
        <!--  最大日志保留时间  -->
        <maxHistory>7</maxHistory>
        <!--  最大单个日志文件大小  -->
        <maxFileSize>10MB</maxFileSize>
    </rollingPolicy>
</appender>

<!--  指定日志输出级别，以及启用的Appender，这里就使用了我们上面的ConsoleAppender  -->
<root level="INFO">
    <appender-ref ref="CONSOLE"/>
    <appender-ref ref="FILE"/>
</root>
```

配置完成后，我们可以看到日志文件也能自动生成了

这里需要提及的是MDC机制，Logback内置的日志字段还是比较少，如果我们需要打印有关业务的更多的内容，包括自定义的一些数据，需要借助logback MDC机制，MDC为“Mapped Diagnostic Context”（映射诊断上下文），即将一些运行时的上下文数据通过logback打印出来；此时我们需要借助org.sl4j.MDC类。

比如我们现在需要记录是哪个用户访问我们网站的日志，只要是此用户访问我们网站，都会在日志中携带该用户的ID，我们希望每条日志中都携带这样一段信息文本，而官方提供的字段无法实现此功能，这时就需要使用MDC机制：             

```java
@ResponseBody
@GetMapping("/test")
public User test(HttpServletRequest request){
   MDC.put("reqId", request.getSession().getId());
   log.info("用户访问了一次测试数据");
   return mapper.findUserById(1);
}
```

通过这种方式，我们就可以向日志中传入自定义参数了，我们日志中添加这样一个占位符`%X{键值}`，名字保持一致：  

```xml
%clr([%X{reqId}]){faint} 
```

这样当我们向MDC中添加信息后，只要是当前线程（本质是ThreadLocal实现）下输出的日志，都会自动替换占位符。

## 3.多环境配置

在日常开发中，我们项目会有多个环境。例如开发环境（develop）也就是我们研发过程中疯狂敲代码修BUG阶段，生产环境（production ）项目开发得差不多了，可以放在服务器上跑了。不同的环境下，可能我们的配置文件也存在不同，但是我们不可能切换环境的时候又去重新写一次配置文件，所以我们可以将多个环境的配置文件提前写好，进行自由切换。

由于SpringBoot只会读取`application.properties`或是`application.yml`文件，那么怎么才能实现自由切换呢？SpringBoot给我们提供了一种方式，我们可以通过配置文件指定：                 

```yaml
spring:
  profiles:
    active: dev
```

接着我们分别创建两个环境的配置文件，`application-dev.yml`和`application-prod.yml`分别表示开发环境和生产环境的配置文件，比如开发环境我们使用的服务器端口为8080，而生产环境下可能就需要设置为80或是443端口，那么这个时候就需要不同环境下的配置文件进行区分：                

```yaml
server:
  port: 8080                    
```

```yaml
server:
  port: 80
```

这样我们就可以灵活切换生产环境和开发环境下的配置文件了。

SpringBoot自带的Logback日志系统也是支持多环境配置的，比如我们想在开发环境下输出日志到控制台，而生产环境下只需要输出到文件即可，这时就需要进行环境配置：                

```xml
<springProfile name="dev">
    <root level="INFO">
        <appender-ref ref="CONSOLE"/>
        <appender-ref ref="FILE"/>
    </root>
</springProfile>

<springProfile name="prod">
    <root level="INFO">
        <appender-ref ref="FILE"/>
    </root>
</springProfile>
```

注意`springProfile`是区分大小写的！

那如果我们希望生产环境中不要打包开发环境下的配置文件呢，我们目前虽然可以切换开发环境，但是打包的时候依然是所有配置文件全部打包，这样总感觉还欠缺一点完美，因此，打包的问题就只能找Maven解决了，Maven也可以设置多环境：             

```xml
<!--分别设置开发，生产环境-->
<profiles>
    <!-- 开发环境 -->
    <profile>
        <id>dev</id>
        <activation>
            <activeByDefault>true</activeByDefault>
        </activation>
        <properties>
            <environment>dev</environment>
        </properties>
    </profile>
    <!-- 生产环境 -->
    <profile>
        <id>prod</id>
        <activation>
            <activeByDefault>false</activeByDefault>
        </activation>
        <properties>
            <environment>prod</environment>
        </properties>
    </profile>
</profiles>
```

接着，我们需要根据环境的不同，排除其他环境的配置文件，在build中写入：                

```xml
<resources>
<!--排除配置文件-->
    <resource>
        <directory>src/main/resources</directory>
        <!--先排除所有的配置文件-->
        <excludes>
            <!--使用通配符，当然可以定义多个exclude标签进行排除-->
            <exclude>application*.yml</exclude>
        </excludes>
    </resource>

    <!--根据激活条件引入打包所需的配置和文件-->
    <resource>
        <directory>src/main/resources</directory>
        <!--引入所需环境的配置文件-->
        <filtering>true</filtering>
        <includes>
            <include>application.yml</include>
            <!--根据maven选择环境导入配置文件-->
            <include>application-${environment}.yml</include>
        </includes>
    </resource>
</resources>
```

接着，我们可以直接将Maven中的`environment`属性，传递给SpringBoot的配置文件，在构建时替换为对应的值：         

```yaml
spring:
  profiles:
    active: '@environment@'  #注意YAML配置文件需要加单引号，否则会报错
```

这样，根据我们Maven环境的切换，SpringBoot的配置文件也会进行对应的切换。

最后我们打开Maven栏目，就可以自由切换了，直接勾选即可，注意切换环境之后要重新加载一下Maven项目，不然不会生效！

## 4.常用框架

### 4.1接口校验规则

首先导入依赖

~~~xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
</dependency>
~~~

然后使用注解完成校验

~~~java
@Slf4j
@Validated   //首先在Controller上开启接口校验
@Controller
public class TestController {

    ...

    @ResponseBody
    @PostMapping("/submit")
    public String submit(@Length(min = 3) String username,  //使用@Length注解一步到位
                         @Length(min = 10) String password){
        System.out.println(username.substring(3));
        System.out.println(password.substring(2, 10));
        return "请求成功!";
    }
}
~~~

创建一个异常处理Controller

~~~java
@ControllerAdvice
public class ValidationController {

    @ResponseBody
    @ExceptionHandler(ConstraintViolationException.class)
    public String error(ValidationException e){
        return e.getMessage();   //出现异常直接返回消息
    }
}
~~~

除了@Length之外，我们也可以使用其他的接口来实现各种数据校验：

|   验证注解   |                        验证的数据类型                        |                           说明                           |
| :----------: | :----------------------------------------------------------: | :------------------------------------------------------: |
| @AssertFalse |                       Boolean,boolean                        |                      值必须是false                       |
| @AssertTrue  |                       Boolean,boolean                        |                       值必须是true                       |
|   @NotNull   |                           任意类型                           |                       值不能是null                       |
|    @Null     |                           任意类型                           |                       值必须是null                       |
|     @Min     | BigDecimal、BigInteger、byte、short、int、long、double 以及任何Number或CharSequence子类型 |                   大于等于@Min指定的值                   |
|     @Max     |                             同上                             |                   小于等于@Max指定的值                   |
| @DecimalMin  |                             同上                             |         大于等于@DecimalMin指定的值（超高精度）          |
| @DecimalMax  |                             同上                             |         小于等于@DecimalMax指定的值（超高精度）          |
|   @Digits    |                             同上                             |                限制整数位数和小数位数上限                |
|    @Size     |               字符串、Collection、Map、数组等                |       长度在指定区间之内，如字符串长度、集合大小等       |
|    @Past     |       如 java.util.Date, java.util.Calendar 等日期类型       |                    值必须比当前时间早                    |
|   @Future    |                             同上                             |                    值必须比当前时间晚                    |
|  @NotBlank   |                     CharSequence及其子类                     |         值不为空，在比较时会去除字符串的首位空格         |
|   @Length    |                     CharSequence及其子类                     |                  字符串长度在指定区间内                  |
|  @NotEmpty   |         CharSequence及其子类、Collection、Map、数组          | 值不为null且长度不为空（字符串长度不为0，集合大小不为0） |
|    @Range    | BigDecimal、BigInteger、CharSequence、byte、short、int、long 以及原子类型和包装类型 |                      值在指定区间内                      |
|    @Email    |                     CharSequence及其子类                     |                     值必须是邮件格式                     |
|   @Pattern   |                     CharSequence及其子类                     |               值需要与指定的正则表达式匹配               |
|    @Valid    |                        任何非原子类型                        |                     用于验证对象属性                     |

如果是封装成对象一起处理：

~~~java
@Data
public class Account {
    String username;
    String password;
}
~~~

只需加入一个注解@Valid

~~~java
@ResponseBody
@PostMapping("/submit")  //在参数上添加@Valid注解表示需要验证
public String submit(@Valid Account account){
    System.out.println(account.getUsername().substring(3));
    System.out.println(account.getPassword().substring(2, 10));
    return "请求成功!";
}
~~~

然后在实体类对应字段加上注解即可

~~~java
@Data
public class Account {
    @Length(min = 3)   //只需要在对应的字段上添加校验的注解即可
    String username;
    @Length(min = 10)
    String password;
}
~~~

再整个异常处理

~~~java
@ResponseBody
@ExceptionHandler({ConstraintViolationException.class, MethodArgumentNotValidException.class})
public String error(Exception e){
    if(e instanceof ConstraintViolationException exception) {
        return exception.getMessage();
    } else if(e instanceof MethodArgumentNotValidException exception){
        if (exception.getFieldError() == null) return "未知错误";
        return exception.getFieldError().getDefaultMessage();
    }
    return "未知错误";
}
~~~

### 4.2接口文档生成

Swagger的主要功能如下：

- 支持 API 自动生成同步的在线文档：使用 Swagger 后可以直接通过代码生成文档，不再需要自己手动编写接口文档了，对程序员来说非常方便，可以节约写文档的时间去学习新技术。
- 提供 Web 页面在线测试 API：光有文档还不够，Swagger 生成的文档还支持在线测试。参数和格式都定好了，直接在界面上输入参数对应的值即可在线测试接口。

结合Spring框架（Spring-doc，官网：https://springdoc.org/），Swagger可以很轻松地利用注解以及扫描机制，来快速生成在线文档，以实现当我们项目启动之后，前端开发人员就可以打开Swagger提供的前端页面，查看和测试接口。依赖如下：                 

```xml
<dependency>
    <groupId>org.springdoc</groupId>
    <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
    <version>2.1.0</version>
</dependency>
```

项目启动之后，我们可以直接访问：http://localhost:8080/swagger-ui/index.html，就能看到我们的开发文档了：

![image-20230717155121213](https://oss.itbaima.cn/internal/markdown/2023/07/17/yb68Oolm1Xp5qFU.png)

可以看到这个开发文档中自动包含了我们定义的接口，并且还有对应的实体类也放在了下面。这个页面不仅仅是展示接口，也可以直接在上面进行调试：

![image-20230717155400761](https://oss.itbaima.cn/internal/markdown/2023/07/17/whLprBimgTqWxFR.png)

这就非常方便了，不仅前端人员可以快速查询接口定义，我们自己也可以在线进行接口测试，直接抛弃PostMan之类的软件了。

虽然Swagger的UI界面已经可以很好地展示后端提供的接口信息了，但是非常的混乱，我们来看看如何配置接口的一些描述信息。首先我们的页面肯定要展示一下这个文档的一些信息，只需要一个Bean就能搞定：                 

```java
@Bean
public OpenAPI springDocOpenAPI() {
        return new OpenAPI().info(new Info()
                        .title("图书管理系统 - 在线API接口文档")   //设置API文档网站标题
                        .description("这是一个图书管理系统的后端API文档，欢迎前端人员查阅！") //网站介绍
                        .version("2.0")   //当前API版本
                        .license(new License().name("我的B站个人主页")  //遵循的协议，这里拿来写其他的也行
                                .url("https://space.bilibili.com/37737161")));
}
```

这样我们的页面中就会展示自定义的文本信息了：

![image-20230717165850714](https://oss.itbaima.cn/internal/markdown/2023/07/17/ZHqL7UsermIbipv.png)

接着我们来看看如何为一个Controller编写API描述信息：                      

```java
//使用@Tag注解来添加Controller描述信息
@Tag(name = "账户验证相关", description = "包括用户登录、注册、验证码请求等操作。")
public class TestController {
	...
}
```

我们可以直接在类名称上面添加`@Tag`注解，并填写相关信息，来为当前的Controller设置描述信息。接着我们可以为所有的请求映射配置描述信息：                    

```java
@ApiResponses({
       @ApiResponse(responseCode = "200", description = "测试成功"),
       @ApiResponse(responseCode = "500", description = "测试失败")   //不同返回状态码描述
})
@Operation(summary = "请求用户数据测试接口")   //接口功能描述
@ResponseBody
@GetMapping("/hello")
//请求参数描述和样例
public String hello(@Parameter(description = "测试文本数据", example = "KFCvivo50") @RequestParam String text) {
    return "Hello World";
}
```

对于那些不需要展示在文档中的接口，我们也可以将其忽略掉：                  

```java
@Hidden
@ResponseBody
@GetMapping("/hello")
public String hello() {
    return "Hello World";
}
```

对于实体类，我们也可以编写对应的API接口文档：                

```java
@Data
@Schema(description = "用户信息实体类")
public class User {
    @Schema(description = "用户编号")
    int id;
    @Schema(description = "用户名称")
    String name;
    @Schema(description = "用户邮箱")
    String email;
    @Schema(description = "用户密码")
    String password;
}
```

这样，我们就可以在文档中查看实体类简介以及各个属性的介绍了。

不过，这种文档只适合在开发环境下生成，如果是生产环境，我们需要关闭文档：                

```java
springdoc:
  api-docs:
    enabled: false
```

这样就可以关闭了

## 5.数据交互

### 5.1JPA快速上手

导入依赖

~~~xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
~~~

接着可以创建实体类

~~~java
@Data
@Entity   //表示这个类是一个实体类
@Table(name = "account")    //对应的数据库中表名称
public class Account {

    @GeneratedValue(strategy = GenerationType.IDENTITY)   //生成策略，这里配置为自增
    @Column(name = "id")    //对应表中id这一列
    @Id     //此属性为主键
    int id;

    @Column(name = "username")   //对应表中username这一列
    String username;

    @Column(name = "password")   //对应表中password这一列
    String password;
}
~~~

然后修改一下yml文件，把日志打印周开

~~~yaml
spring:
  jpa:
    #开启SQL语句执行日志信息
    show-sql: true
    hibernate:
      #配置为检查数据库表结构，没有时会自动创建
      ddl-auto: update
~~~

`ddl-auto`属性用于设置自动表定义，可以实现自动在数据库中为我们创建一个表，表的结构会根据我们定义的实体类决定，它有以下几种：

- `none`: 不执行任何操作，数据库表结构需要手动创建。
- `create`: 框架在每次运行时都会删除所有表，并重新创建。
- `create-drop`: 框架在每次运行时都会删除所有表，然后再创建，但在程序结束时会再次删除所有表。
- `update`: 框架会检查数据库表结构，如果与实体类定义不匹配，则会做相应的修改，以保持它们的一致性。
- `validate`: 框架会检查数据库表结构与实体类定义是否匹配，如果不匹配，则会抛出异常。

这个配置项的作用是为了避免手动管理数据库表结构，使开发者可以更方便地进行开发和测试，但在生产环境中，更推荐使用数据库迁移工具来管理表结构的变更。

接着看如何访问表，首先需要新建一个Repository实现类

~~~java
@Repository
//前者是对象类，后者是主键的类型
public interface AccountRepository extends JpaRepository<Account, Integer> {
}
~~~

然后测试即可

~~~java
@Resource
AccountRepository repository;

@Test
void contextLoads() {
    Account account = new Account();
    account.setUsername("小红");
    account.setPassword("1234567");
    System.out.println(repository.save(account).getId());   //使用save来快速插入数据，并且会返回插入的对象，如果存在自增ID，对象的自增id属性会自动被赋值，这就很方便了
}
~~~

### 5.2方法名称拼接SQL

| 属性               | 拼接方法名称示例                                            | 执行的语句                                                   |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Distinct           | findDistinctByLastnameAndFirstname                          | select distinct … where x.lastname = ?1 and x.firstname = ?2 |
| And                | findByLastnameAndFirstname                                  | … where x.lastname = ?1 and x.firstname = ?2                 |
| Or                 | findByLastnameOrFirstname                                   | … where x.lastname = ?1 or x.firstname = ?2                  |
| Is，Equals         | findByFirstname`,`findByFirstnameIs`,`findByFirstnameEquals | … where x.firstname = ?1                                     |
| Between            | findByStartDateBetween                                      | … where x.startDate between ?1 and ?2                        |
| LessThan           | findByAgeLessThan                                           | … where x.age < ?1                                           |
| LessThanEqual      | findByAgeLessThanEqual                                      | … where x.age <= ?1                                          |
| GreaterThan        | findByAgeGreaterThan                                        | … where x.age > ?1                                           |
| GreaterThanEqual   | findByAgeGreaterThanEqual                                   | … where x.age >= ?1                                          |
| After              | findByStartDateAfter                                        | … where x.startDate > ?1                                     |
| Before             | findByStartDateBefore                                       | … where x.startDate < ?1                                     |
| IsNull，Null       | findByAge(Is)Null                                           | … where x.age is null                                        |
| IsNotNull，NotNull | findByAge(Is)NotNull                                        | … where x.age not null                                       |
| Like               | findByFirstnameLike                                         | … where x.firstname like ?1                                  |
| NotLike            | findByFirstnameNotLike                                      | … where x.firstname not like ?1                              |
| StartingWith       | findByFirstnameStartingWith                                 | … where x.firstname like ?1（参数与附加`%`绑定）             |
| EndingWith         | findByFirstnameEndingWith                                   | … where x.firstname like ?1（参数与前缀`%`绑定）             |
| Containing         | findByFirstnameContaining                                   | … where x.firstname like ?1（参数绑定以`%`包装）             |
| OrderBy            | findByAgeOrderByLastnameDesc                                | … where x.age = ?1 order by x.lastname desc                  |
| Not                | findByLastnameNot                                           | … where x.lastname <> ?1                                     |
| In                 | findByAgeIn(Collection<Age> ages)                           | … where x.age in ?1                                          |
| NotIn              | findByAgeNotIn(Collection<Age> ages)                        | … where x.age not in ?1                                      |
| True               | findByActiveTrue                                            | … where x.active = true                                      |
| False              | findByActiveFalse                                           | … where x.active = false                                     |
| IgnoreCase         | findByFirstnameIgnoreCase                                   | … where UPPER(x.firstname) = UPPER(?1)                       |

比如我们想要实现根据用户名模糊匹配查找用户：                    

```java
@Repository
public interface AccountRepository extends JpaRepository<Account, Integer> {
    //按照表中的规则进行名称拼接，不用刻意去记，IDEA会有提示
    List<Account> findAllByUsernameLike(String str);
}
```

测试一下

~~~java
@Test
void contextLoads() {
    repository.findAllByUsernameLike("%明%").forEach(System.out::println);
}
~~~

### 5.3关联查询

**一对一**

定义数据库表

~~~java
@Data
@Entity
@Table(name = "account_detail")
public class AccountDetail {

    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    int id;

    @Column(name = "address")
    String address;

    @Column(name = "email")
    String email;

    @Column(name = "phone")
    String phone;

    @Column(name = "real_name")
    String realName;
}
~~~

然后指定关联列

~~~java
@Data
@Entity
@Table(name = "account")
public class Account {
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    int id;

    @Column(name = "username")
    String username;

    @Column(name = "password")
    String password;

    @JoinColumn(name = "detail_id")   //指定存储外键的字段名称
    @OneToOne    //声明为一对一关系
    AccountDetail detail;

}
~~~

然后可以快乐的调用方法

~~~java
    @Test
    void pageAccount() {
        repository.findById(1).ifPresent(System.out::println);
    }

~~~

那么我们是否也可以在添加数据时，利用实体类之间的关联信息，一次性添加两张表的数据呢？可以，但是我们需要稍微修改一下级联关联操作设定：                 

```java
@JoinColumn(name = "detail_id")
@OneToOne(cascade = CascadeType.ALL) //设置关联操作为ALL
AccountDetail detail;
```

- ALL：所有操作都进行关联操作
- PERSIST：插入操作时才进行关联操作
- REMOVE：删除操作时才进行关联操作
- MERGE：修改操作时才进行关联操作

即可测试

~~~java
@Test
void addAccount(){
    Account account = new Account();
    account.setUsername("Nike");
    account.setPassword("123456");
    AccountDetail detail = new AccountDetail();
    detail.setAddress("重庆市渝中区解放碑");
    detail.setPhone("1234567890");
    detail.setEmail("73281937@qq.com");
    detail.setRealName("张三");
  	account.setDetail(detail);
    account = repository.save(account);
    System.out.println("插入时，自动生成的主键ID为："+account.getId()+"，外键ID为："+account.getDetail().getId());
}
~~~

**一对多**

接着我们来看一对多关联，比如每个用户的成绩信息：                

```java
@JoinColumn(name = "uid")  //注意这里的name指的是Score表中的uid字段对应的就是当前的主键，会将uid外键设置为当前的主键
@OneToMany(fetch = FetchType.LAZY, cascade = CascadeType.REMOVE)   //在移除Account时，一并移除所有的成绩信息，依然使用懒加载
List<Score> scoreList;              
```

```java
@Data
@Entity
@Table(name = "users_score")   //成绩表，注意只存成绩，不存学科信息，学科信息id做外键
public class Score {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    @Id
    int id;

    @OneToOne   //一对一对应到学科上
    @JoinColumn(name = "cid")
    Subject subject;

    @Column(name = "socre")
    double score;

    @Column(name = "uid")
    int uid;
}               
```

```java
@Data
@Entity
@Table(name = "subjects")   //学科信息表
public class Subject {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cid")
    @Id
    int cid;

    @Column(name = "name")
    String name;

    @Column(name = "teacher")
    String teacher;

    @Column(name = "time")
    int time;
}
```

在数据库中填写相应数据，接着我们就可以查询用户的成绩信息了：                

```java
@Transactional
@Test
void test() {
    repository.findById(1).ifPresent(account -> {
        account.getScoreList().forEach(System.out::println);
    });
}
```

成功得到用户所有的成绩信息，包括得分和学科信息。

**多对一**

同样的，我们还可以将对应成绩中的教师信息单独分出一张表存储，并建立多对一的关系，因为多门课程可能由同一个老师教授：               

```java
@ManyToOne(fetch = FetchType.LAZY)
@JoinColumn(name = "tid")   //存储教师ID的字段，和一对一是一样的，也会当前表中创个外键
Teacher teacher;
```

接着就是教师实体类了：                   

```java
@Data
@Entity
@Table(name = "teachers")
public class Teacher {

    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    int id;

    @Column(name = "name")
    String name;

    @Column(name = "sex")
    String sex;
}
```

最后我们再进行一下测试：              

```java
@Transactional
@Test
void test() {
    repository.findById(3).ifPresent(account -> {
        account.getScoreList().forEach(score -> {
            System.out.println("课程名称："+score.getSubject().getName());
            System.out.println("得分："+score.getScore());
            System.out.println("任课教师："+score.getSubject().getTeacher().getName());
        });
    });
}
```

成功得到多对一的教师信息。

**多对多**

最后我们再来看最复杂的情况，现在我们一门课程可以由多个老师教授，而一个老师也可以教授多个课程，那么这种情况就是很明显的多对多场景，现在又该如何定义呢？我们可以像之前一样，插入一张中间表表示教授关系，这个表中专门存储哪个老师教哪个科目：         

```java
@ManyToMany(fetch = FetchType.LAZY)   //多对多场景
@JoinTable(name = "teach_relation",     //多对多中间关联表
        joinColumns = @JoinColumn(name = "cid"),    //当前实体主键在关联表中的字段名称
        inverseJoinColumns = @JoinColumn(name = "tid")   //教师实体主键在关联表中的字段名称
)
List<Teacher> teacher;
```

接着，JPA会自动创建一张中间表，并自动设置外键，我们就可以将多对多关联信息编写在其中了。

### 5.4JPQL自定义SQL语言

例如实现根据用户名来修改密码

~~~java
@Repository
public interface AccountRepository extends JpaRepository<Account, Integer> {

    @Transactional    //DML操作需要事务环境，可以不在这里声明，但是调用时一定要处于事务环境下
    @Modifying     //表示这是一个DML操作
    @Query("update Account set password = ?2 where id = ?1") //这里操作的是一个实体类对应的表，参数使用?代表，后面接第n个参数
    int updatePasswordById(int id, String newPassword);
}
~~~

或者使用原生SQL

~~~java
@Transactional
@Modifying
@Query(value = "update users set password = :pwd where username = :name", nativeQuery = true) //使用原生SQL，和Mybatis一样，这里使用 :名称 表示参数，当然也可以继续用上面那种方式。
int updatePasswordByUsername(@Param("name") String username,   //我们可以使用@Param指定名称
                             @Param("pwd") String newPassword);
~~~

### 5.5MyBatis-Plus

#### 快速上手

跟之前一样，还是添加依赖：                 

```xml
<dependency>
     <groupId>com.baomidou</groupId>
     <artifactId>mybatis-plus-boot-starter</artifactId>
     <version>3.5.3.1</version>
</dependency>
<dependency>
     <groupId>com.mysql</groupId>
     <artifactId>mysql-connector-j</artifactId>
</dependency>
```

配置文件依然只需要配置数据源即可：                  

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/test
    username: root
    password: 123456
    driver-class-name: com.mysql.cj.jdbc.Driver
```

然后依然是实体类，可以直接映射到数据库中的表：                     

```java
@Data
@TableName("user")  //对应的表名
public class User {
    @TableId(type = IdType.AUTO)   //对应的主键
    int id;
    @TableField("name")   //对应的字段
    String name;
    @TableField("email")
    String email;
    @TableField("password")
    String password;
}
```

接着，我们就可以编写一个Mapper来操作了：

```java
@Mapper
public interface UserMapper extends BaseMapper<User> {
  	//使用方式与JPA极其相似，同样是继承一个基础的模版Mapper
  	//这个模版里面提供了预设的大量方法直接使用，跟JPA如出一辙
}
```

这里我们就来写一个简单测试用例：

```java
@SpringBootTest
class DemoApplicationTests {

    @Resource
    UserMapper mapper;

    @Test
    void contextLoads() {
        System.out.println(mapper.selectById(1));  //同样可以直接selectById，非常快速方便
    }
}
```

可以看到这个Mapper提供的方法还是很丰富的：

![image-20230721133315171](https://oss.itbaima.cn/internal/markdown/2023/07/21/R7fhN5UtAOPFe4M.png)

后续的板块我们将详细介绍它的使用方式。

#### 条件构造器

对于一些复杂查询的情况，MybatisPlus支持我们自己构造QueryWrapper用于复杂条件查询：          

```java
@Test
void contextLoads() {
    QueryWrapper<User> wrapper = new QueryWrapper<>();    //复杂查询可以使用QueryWrapper来完成
  	wrapper
            .select("id", "name", "email", "password")    //可以自定义选择哪些字段
            .ge("id", 2)     			//选择判断id大于等于1的所有数据
            .orderByDesc("id");   //根据id字段进行降序排序
    System.out.println(mapper.selectList(wrapper));   //Mapper同样支持使用QueryWrapper进行查询
}
```

通过使用上面的QueryWrapper对象进行查询，也就等价于下面的SQL语句：            

```sql
select id,name,email,password from user where id >= 2 order by id desc
```

我们可以在配置中开启SQL日志打印：           

```yaml
mybatis-plus:
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```

最后得到的结果如下：

![image-20230721160951500](https://oss.itbaima.cn/internal/markdown/2023/07/21/FxOfrnERhVPi8tu.png)

有些时候我们遇到需要批处理的情况，也可以直接使用批处理操作：                

```java
@Test
void contextLoads() {
    //支持批处理操作，我们可以一次性删除多个指定ID的用户
    int count = mapper.deleteBatchIds(List.of(1, 3));
    System.out.println(count);
}
```

![image-20230721190139253](https://oss.itbaima.cn/internal/markdown/2023/07/21/lwaJUF3g2opbWZG.png)

我们也可以快速进行分页查询操作，不过在执行前我们需要先配置一下：                     

```java
@Configuration
public class MybatisConfiguration {
    @Bean
    public MybatisPlusInterceptor paginationInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
      	//添加分页拦截器到MybatisPlusInterceptor中
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
        return interceptor;
    }
}
```

这样我们就可以愉快地使用分页功能了：          

```java
@Test
void contextLoads() {
    //这里我们将用户表分2页，并获取第一页的数据
    Page<User> page = mapper.selectPage(Page.of(1, 2), Wrappers.emptyWrapper());
    System.out.println(page.getRecords());   //获取分页之后的数据
}
```

![image-20230721185519292](https://oss.itbaima.cn/internal/markdown/2023/07/21/XMPLWB3N6VpHUkG.png)

对于数据更新操作，我们也可以使用UpdateWrapper非常方便的来完成：                      

```java
@Test
void contextLoads() {
    UpdateWrapper<User> wrapper = new UpdateWrapper<>();
    wrapper
            .set("name", "lbw")
            .eq("id", 1);
    System.out.println(mapper.update(null, wrapper));
}
```

这样就可以快速完成更新操作了：

![image-20230721162409308](https://oss.itbaima.cn/internal/markdown/2023/07/21/W1e8fFuUwSpi7Cg.png)

QueryWrapper和UpdateWrapper还有专门支持Java 8新增的Lambda表达式的特殊实现，可以直接以函数式的形式进行编写，使用方法是一样的，这里简单演示几个：                    

```java
@Test
void contextLoads() {
        LambdaQueryWrapper<User> wrapper = Wrappers
                .<User>lambdaQuery()
                .eq(User::getId, 2)   //比如我们需要选择id为2的用户，前面传入方法引用，后面比的值
                .select(User::getName, User::getId);   //比如我们只需要选择name和id，那就传入对应的get方法引用
        System.out.println(mapper.selectOne(wrapper));
}
```

不过感觉可读性似乎没有不用Lambda高啊。

#### 接口基本操作

虽然使用MybatisPlus提供的BaseMapper已经很方便了，但是我们的业务中，实际上很多时候也是一样的工作，都是去简单调用底层的Mapper做一个很简单的事情，那么能不能干脆把Service也给弄个模版？MybatisPlus为我们提供了很方便的CRUD接口，直接实现了各种业务中会用到的增删改查操作。

我们只需要继承即可：

```java
@Service
public interface UserService extends IService<User> {
  	//除了继承模版，我们也可以把它当成普通Service添加自己需要的方法
}
```

接着我们还需要编写一个实现类，这个实现类就是UserService的实现：                   

```java
@Service   //需要继承ServiceImpl才能实现那些默认的CRUD方法
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements UserService {
}
```

使用起来也很方便，整合了超多方法：

![image-20230721181359616](https://oss.itbaima.cn/internal/markdown/2023/07/21/l5Vkb9dgtJcyL4R.png)

比如我们想批量插入一组用户数据到数据库中：               

```java
@Test
void contextLoads() {
    List<User> users = List.of(new User("xxx"), new User("yyy"));
  	//预设方法中已经支持批量保存了，这相比我们直接用for效率高不少
    service.saveBatch(users);
}
```

还有更加方便快捷的保存或更新操作，当数据不存在时（通过主键ID判断）则插入新数据，否则就更新数据：                      

```java
@Test
void contextLoads() {
    service.saveOrUpdate(new User("aaa"));
}
```

我们也可以直接使用Service来进行链式查询，写法非常舒服：                       

```java
@Test
void contextLoads() {
    User one = service.query().eq("id", 1).one();
    System.out.println(one);
}
```

#### 新版代码生成器

最后我们再来隆重介绍一下MybatisPlus的代码生成器，这个东西可谓是拯救了千千万万学子的毕设啊。

它能够根据数据库做到代码的一键生成，能做到什么程度呢？

![image-20230721200757985](https://oss.itbaima.cn/internal/markdown/2023/07/21/lGT4g5Y6Heqavsw.png)

你没看错，整个项目从Mapper到Controller，所有的东西全部都给你生成好了，你只管把需要补充的业务给写了就行，这是真正的把饭给喂到你嘴边的行为，是广大学子的毕设大杀器。

那么我们就来看看，这玩意怎么去用的，首先我们需要先把整个项目的数据库给创建好，创建好之后，我们继续下一步，这里我们从头开始创建一个项目，感受一下它的强大，首先创建一个普通的SpringBoot项目：

![image-20230721202019230](https://oss.itbaima.cn/internal/markdown/2023/07/21/bIZ9D2cA7XsgSoU.png)

接着我们导入一会需要用到的依赖：                       

```xml
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-boot-starter</artifactId>
    <version>3.5.3.1</version>
</dependency>
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-generator</artifactId>
    <version>3.5.3.1</version>
</dependency>
<dependency>
    <groupId>org.apache.velocity</groupId>
    <artifactId>velocity-engine-core</artifactId>
    <version>2.3</version>
</dependency>
```

然后再配置一下数据源：                   

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/test
    username: root
    password: 123456
    driver-class-name: com.mysql.cj.jdbc.Driver
```

接着我们就可以开始编写自动生成脚本了，这里依然选择测试类，用到`FastAutoGenerator`作为生成器：                   

```java
		@Test
    void contextLoads() {
        FastAutoGenerator
          			//首先使用create来配置数据库链接信息
                .create(new DataSourceConfig.Builder(dataSource))
                .execute();
    }
```

接着我们配置一下全局设置，这些会影响一会生成的代码：                 

```java
@Test
void contextLoads() {
    FastAutoGenerator
            .create(new DataSourceConfig.Builder(dataSource))
            .globalConfig(builder -> {
                builder.author("lbw");              //作者信息，一会会变成注释
                builder.commentDate("2024-01-01");  //日期信息，一会会变成注释
                builder.outputDir("src/main/java"); //输出目录设置为当前项目的目录
            })
            .execute();
}
```

然后是打包设置，也就是项目的生成的包等等，这里简单配置一下：                  

```java
@Test
void contextLoads() {
    FastAutoGenerator
            ...
      			//打包设置，这里设置一下包名就行，注意跟我们项目包名设置为一致的
      			.packageConfig(builder -> builder.parent("com.example"))
      			.strategyConfig(builder -> {
                    //设置为所有Mapper添加@Mapper注解
                    builder
                            .mapperBuilder()
                            .mapperAnnotation(Mapper.class)
                            .build();
            })
            .execute();
}
```

接着我们就可以直接执行了这个脚本了：

![image-20230721203819514](https://oss.itbaima.cn/internal/markdown/2023/07/21/SdDRqZPnNrkeKjG.png)

现在，可以看到我们的项目中已经出现自动生成代码了：

![image-20230721204011913](https://oss.itbaima.cn/internal/markdown/2023/07/21/pKMnwFZEOBmLXDy.png)

我们也可以直接运行这个项目：

![image-20230721210417345](https://oss.itbaima.cn/internal/markdown/2023/07/21/CEdRz5wgaoxUjFJ.png)

速度可以说是非常之快，一个项目模版就搭建完成了，我们只需要接着写业务就可以了，当然如果各位小伙伴需要更多定制化的话，可以在官网查看其他的配置：https://baomidou.com/pages/981406/

对于一些有特殊要求的用户来说，我们希望能够以自己的模版来进行生产，怎么才能修改它自动生成的代码模版呢，我们可以直接找到`mybatis-plus-generator`的源码：

![image-20230721204530505](https://oss.itbaima.cn/internal/markdown/2023/07/21/lxaBgGPubOkptCT.png)

生成模版都在在这个里面有写，我们要做的就是去修改这些模版，变成我们自己希望的样子，由于默认的模版解析引擎为Velocity，我们需要复制以`.vm`结尾的文件到`resource`随便一个目录中，然后随便改：

![image-20230721210716832](https://oss.itbaima.cn/internal/markdown/2023/07/21/gZlbG9JDIa3kSMO.png)

接着我们配置一下模版：           

```java
@Test
void contextLoads() {
    FastAutoGenerator
            ...
      			.strategyConfig(builder -> {
                builder
                        .mapperBuilder()
                        .enableFileOverride()   //开启文件重写，自动覆盖新的
                        .mapperAnnotation(Mapper.class)
                        .build();
            })
            .templateConfig(builder -> {
                builder.mapper("/template/mapper.java.vm");
            })
            .execute();
}
```

这样，新生成的代码中就是按照我们自己的模版来定义了:

![image-20230721211002961](https://oss.itbaima.cn/internal/markdown/2023/07/21/K6DufSwG3hdqPsr.png)