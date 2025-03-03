# Spring

## 1.IoC容器基础

### 1.1IoC理论基础

IoC即控制反转，比如需要一个接口实现，无需关心具体使用哪个接口，而是让配置文件决定，只管调用就行

~~~java
public static void main(String[] args) {
		A a = new A();
  	a.test(IoC.getBean(Service.class));   //瞎编的一个容器类，但是是那个意思
  	//比如现在在IoC容器中管理的Service的实现是B，那么我们从里面拿到的Service实现就是B
}

class A{
    private List<Service> list;   //一律使用Service，具体实现由IoC容器提供
    public Service test(Service b){
        return null;
    }
}

interface Service{ }   //使用Service做一个顶层抽象

class B implements Service{}  //B依然是具体实现类，并交给IoC容器管理
~~~

如果要修改B为D，直接改就行，不会报错

~~~java
interface Service{ }

class D implements Service{}   //现在实现类变成了D，但是之前的代码并不会报错
~~~

###  1.2搭建Spring

首先需要配置上下文应用文件application.xml，在其中写入各种bean‘

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
        https://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean name ="student" class="org.example.entity.Student"></bean>
</beans>
~~~

然后加载配置文件，并直接调用即可，可用通过name或者class调用，`class`参数指定了Spring容器中bean的实现类。在代码中，它指定了`org.example.entity.Student`类作为bean的实现类。  

~~~java
public static void main(String[] args) {
        //加载配置文件管理bean
        ApplicationContext context = new ClassPathXmlApplicationContext("application.xml");
        Student student = context.getBean(Student.class);
        Student student2 = (Student) context.getBean("student");
}
~~~

### 1.3Bean注册于配置

注册Bean的配置，同时对于接口来说如果没有声明则会直接调用声明的子类

~~~xml
    <bean name ="student" class="org.example.entity.Student"></bean>
    <bean class="org.example.entity.ArtStudent"></bean>
    <bean class="org.example.entity.SportStudent"></bean>
~~~

也可以通过命名来区分同一个Bean，通过name来使用，name**不可重复**

~~~xml
<bean name="art" class="com.test.bean.ArtStudent"/>
<bean name="sport" class="com.test.bean.SportStudent"/>
~~~

对于Bean来说，分为**单例模式**和**原型模式**

- 单例模式：在容器加载的时候就会被创建，知只要容器没有被销毁就一直存在，只存在一个
- 原型模式：每次获取Bean的时候才会创建，相当于每次都new一个

~~~xml
    <bean name ="student" class="org.example.entity.Student"></bean>
    <bean class="org.example.entity.ArtStudent" scope="prototype"></bean>
    <bean class="org.example.entity.SportStudent" scope="singleton"></bean>
~~~

如果对于单例模式想要在获取的时候才创建，需要引入新参数

~~~xml
<bean class="com.test.bean.Student" lazy-init="true"/>
~~~

同样的，如果对Bean的加载顺序有要求，比如Student必须要在Teacher之后加载

~~~xml
<bean name="teacher" class="com.test.bean.Teacher"/>
<bean name="student" class="com.test.bean.Student" depends-on="teacher"/>
~~~

### 1.4依赖注入

**通过setter注入**

对于类的某个属性如果发生更改，无需修改每个类，而是只修改配置文件即可

1. 将属性作为Bean传入，指定路径
2. 将类作为依赖注入的容器，配置相关属性，其中ref注入其他Bean，value直接注入值

~~~xml
    <bean name="teacher" class="org.example.entity.ArtTeacher"/>

    <bean name ="student" class="org.example.entity.Student">
        <property name="teacher" ref="teacher"/>
        <property name="name" value="小明"/>
    </bean>
~~~

**通过构造方法注入**

前面说，Bean实际上是由IoC容器进行创建的，如果修改默认的构造方法，配置文件会报错

~~~java
public class Student {
    private final Teacher teacher;public class Student {
    Teacher teacher;
    String name;

    public Student(Teacher teacher, String name) {
        this.teacher = teacher;
        this.name = name;
    }

    }
}
~~~

因此需要指明一个可用的构造方法，展开Bean标签，添加`constructor-arg`标签，几个参数就几个标签

~~~java
<bean name="teacher" class="com.test.bean.ArtTeacher"/>
<bean name="student" class="com.test.bean.Student">
    <constructor-arg name="teacher" ref="teacher"/>
    <constructor-arg name="name" value="小明"/>
</bean>
~~~

对于List，Map等特殊属性，也有对应注入方法

~~~xml
<bean name="student" class="com.test.bean.Student">
  	<!--  对于集合类型，我们可以直接使用标签编辑集合的默认值  -->
    <property name="list">
        <list>
            <value>AAA</value>
            <value>BBB</value>
            <value>CCC</value>
        </list>
    </property>
</bean>
~~~

~~~xml
<bean name="student" class="com.test.bean.Student">
    <property name="map">
        <map>
            <entry key="语文" value="100.0"/>
            <entry key="数学" value="80.0"/>
            <entry key="英语" value="92.5"/>
        </map>
    </property>
</bean>
~~~

### 1.5自动装配

有些时候为了方便，我们也可以开启自动装配。自动装配就是让IoC容器自己去寻找需要填入的值，我们只需要将set方法提供好就可以了，这里需要添加autowire属性：                     

```xml
<bean name="student" class="com.test.bean.Student" autowire="byType"/>
```

`autowire`属性有两个值普通，一个是byName，还有一个是byType，顾名思义，一个是根据类型去寻找合适的Bean自动装配，还有一个是根据名字去找，这样我们就不需要显式指定`property`了。

![image-20221122221936559](https://oss.itbaima.cn/internal/markdown/2022/11/22/QIBRwScq6fu4XDm.png)

此时set方法旁边会出现一个自动装配图标，效果和上面是一样的。

对于使用**构造方法**完成的依赖注入，也支持自动装配，我们只需要将autowire修改为：                     

```xml
<bean name="student" class="com.test.bean.Student" autowire="constructor"/>
```

这样，我们只需要提供一个对应参数的构造方法就可以了（这种情况默认也是byType寻找的）：

![image-20221122230320004](https://oss.itbaima.cn/internal/markdown/2022/11/22/rgl7fXJ2ZKAU8Rd.png)

这样同样可以完成自动注入：

![image-20221122191427776](https://oss.itbaima.cn/internal/markdown/2022/11/22/evKArqDYcIQPCXT.png)

自动化的东西虽然省事，但是太过机械，有些时候，自动装配可能会遇到一些问题，比如出现了下面的情况：

![image-20221122223048820](https://oss.itbaima.cn/internal/markdown/2022/11/22/SQTchJBq4G8NWyC.png)

此时，由于`autowire`的规则为byType，存在两个候选Bean，但是我们其实希望ProgramTeacher这个Bean在任何情况下都不参与到自动装配中，此时我们就可以将它的自动装配候选关闭：                    

```xml
<bean name="teacher" class="com.test.bean.ArtTeacher"/>
<bean name="teacher2" class="com.test.bean.ProgramTeacher" autowire-candidate="false"/>
<bean name="student" class="com.test.bean.Student" autowire="byType"/>
```

当`autowire-candidate`设定false时，这个Bean将不再作为自动装配的候选Bean，此时自动装配候选就只剩下一个唯一的Bean了，报错消失，程序可以正常运行。

除了这种方式，我们也可以设定primary属性，表示这个Bean作为主要的Bean，当出现歧义时，也会优先选择：             

```xml
<bean name="teacher" class="com.test.bean.ArtTeacher" primary="true"/>
<bean name="teacher2" class="com.test.bean.ProgramTeacher"/>
<bean name="student" class="com.test.bean.Student" autowire="byType"/>
```

### 1.6生命周期与继承

除了修改构造方法，我们也可以为Bean指定初始化方法和销毁方法，以便在对象创建和被销毁时执行一些其他的任务：                      

```java
public void init(){
    System.out.println("我是对象初始化时要做的事情！");    
}

public void destroy(){
    System.out.println("我是对象销毁时要做的事情！");
}
```

我们可以通过`init-method`和`destroy-method`来指定：                

```xml
<bean name="student" class="com.test.bean.Student" init-method="init" destroy-method="destroy"/>
```

那么什么时候是初始化，什么时候又是销毁呢？                   

```java
//当容器创建时，默认情况下Bean都是单例的，那么都会在一开始就加载好，对象构造完成后，会执行init-method
ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext("test.xml");
//我们可以调用close方法关闭容器，此时容器内存放的Bean也会被一起销毁，会执行destroy-method
context.close();
```

注意，以上所有只针对单例模式，原型模式只会在创建时启动初始化方法，不会调用销毁方法。

**继承**

依赖注入可以继承，但是子Bean必须具有父Bean的所有属性，也可以**另加**新属性，也可以**覆盖**

~~~xml
<bean name="artStudent" class="com.test.bean.ArtStudent" abstract="true">
    <property name="name" value="小明"/>
    <property name="id" value="1"/>
</bean>
<bean class="com.test.bean.SportStudent" parent="artStudent">
    <property name="id" value="2"/>
</bean>
~~~

如果只希望某个Bean作为配置模板而不使用，则可以只作为继承使用，无法直接获取context.getBean()

~~~xml
<bean name="artStudent" class="com.test.bean.ArtStudent" abstract="true">
    <property name="name" value="小明"/>
</bean>
<bean class="com.test.bean.SportStudent" parent="artStudent"/>
~~~

如果希望所有Bean都使用某种方式配置，可以在最外层Beans中进行配置

~~~xml
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
        https://www.springframework.org/schema/beans/spring-beans.xsd" default-autowire="byName">
~~~

### 1.7注解开发

现在使用AnnotationConfigApplicationContext作为上下文实现，它是注解配置的。

~~~java
ApplicationContext context = new AnnotationConfigApplicationContext();
~~~

创建一个配置类即可

~~~java
@Configuration
public class MainConfiguration {
}

~~~

然后可以添加各种Bean

~~~java
@Configuration
public class MainConfiguration {

    @Bean
    public Teacher teacher() {
        return new ArtTeacher();
    }

    @Bean("student")
    public Student student(Teacher teacher) {
        Student student = new Student();
        student.setTeacher(teacher);
        return new Student();
    }
}
~~~

对于各种特性也可以配置

~~~java
@Bean(name = "", initMethod = "", destroyMethod = "", autowireCandidate = false)
public Student student(){
    return new Student();
}
~~~

以及其他注解配置其他属性

~~~java
@Bean
@Lazy(true)     //对应lazy-init属性
@Scope("prototype")    //对应scope属性
@DependsOn("teacher")    //对应depends-on属性
public Student student(){
    return new Student();
}
~~~

对于使用构造方法或者setter完成依赖注入的Bean，可以直接将其作为参数

~~~java
@Configuration
public class MainConfiguration {
    @Bean
    public Teacher teacher(){
        return new Teacher();
    }

    @Bean
    public Student student(Teacher teacher){
        return new Student(teacher);
    }
}
~~~

同时，我们也可以直接使用注解进行自动装配，更加简单

~~~java
public class Student {
    @Autowired
    Teacher teacher;
    }
~~~

注意配置类中无需setter方法

~~~java
    public Student student() {
        Student student = new Student();
        return new Student();
    }
~~~

对于多个类型相同的Bean，则需要指定一个Bean，使用注解@Qualifier

~~~java
public class Student {
    @Autowired
    @Qualifier("a")   //匹配名称为a的Teacher类型的Bean
    private Teacher teacher;
}
~~~

除了这个注解之外，还有@PostConstruct和@PreDestroy，它们效果和init-method和destroy-method是一样的： 

```java
@PostConstruct
public void init(){
    System.out.println("我是初始化方法");
}

@PreDestroy
public void destroy(){
    System.out.println("我是销毁方法");
}
```

要实现容器自己反射获取构造方法生成对象，对于包下所有注解`@Component`的类都会自动注册Bean

```java
@Component("lbwnb")   //同样可以自己起名字
public class Student {

}
```

要注册这个类的Bean，只需要添加@Component即可，然后配置一下包扫描，容器会对包内所有带@Component的类注册                   

```java
@Configuration
@ComponentScan("com.test.bean")   //包扫描，这样Spring就会去扫描对应包下所有的类
public class MainConfiguration {

}
```

## 2.Spring高级特性

### 2.1Bean Aware

在Spring中提供了一些以Aware结尾的接口，实现了Aware接口的bean在被初始化之后，可以获取相应资源。Aware的中文意思为**感知**。简单来说，他就是一个标识，实现此接口的类会获得某些感知能力，Spring容器会在Bean被加载时，根据类实现的感知接口，会调用类中实现的对应感知方法。

比如BeanNameAware之类的以Aware结尾的接口，这个接口获取的资源就是BeanName：       

```java
@Component
public class Student implements BeanNameAware {   //我们只需要实现这个接口就可以了

    @Override
    public void setBeanName(String name) {   //Bean在加载的时候，容器就会自动调用此方法，将Bean的名称给到我们
        System.out.println("我在加载阶段获得了Bean名字："+name);
    }
}
```

### 2.2任务调度

首先我们来看**异步任务**执行，需要使用Spring异步任务支持，我们需要在配置类上添加`@EnableAsync`注解。            

```java
@EnableAsync
@Configuration
@ComponentScan("com.test.bean")
public class MainConfiguration {
}
```

接着我们只需要在需要异步执行的方法上，添加`@Async`注解即可将此方法标记为异步，当此方法被调用时，会异步执行，也就是新开一个线程执行，而不是在当前线程执行。我们来测试一下：                   

```java
@Component
public class Student {
    public void syncTest() throws InterruptedException {
        System.out.println(Thread.currentThread().getName()+"我是同步执行的方法，开始...");
        Thread.sleep(3000);
        System.out.println("我是同步执行的方法，结束！");
    }

    @Async
    public void asyncTest() throws InterruptedException {
        System.out.println(Thread.currentThread().getName()+"我是异步执行的方法，开始...");
        Thread.sleep(3000);
        System.out.println("我是异步执行的方法，结束！");
    }
}
```

现在我们在主方法中分别调用一下试试看：             

```java
public static void main(String[] args) throws InterruptedException {
    AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(MainConfiguration.class);
    Student student = context.getBean(Student.class);
    student.asyncTest();   //异步执行
    student.syncTest();    //同步执行
}
```

可以看到，我们的任务执行结果为：

![image-20221125153110860](https://oss.itbaima.cn/internal/markdown/2022/11/25/7VKh3dreROJUTcN.png)

很明显，异步执行的任务并不是在当前线程启动的，而是在其他线程启动的，所以说并不会在当前线程阻塞，可以看到马上就开始执行下一行代码，调用同步执行的任务了。

因此，当我们要将Bean的某个方法设计为异步执行时，就可以直接添加这个注解。但是需要注意，添加此注解要求方法的返回值只能是**void**或是**Future**类型才可以。

**定时任务**

Spring中的定时任务是全局性质的，当我们的Spring程序启动后，那么定时任务也就跟着启动了，我们可以在配置类上添加`@EnableScheduling`注解：               

```java
@EnableScheduling
@Configuration
@ComponentScan("com.test.bean")
public class MainConfiguration {
}
```

接着我们可以直接在配置类里面编写定时任务，把我们要做的任务写成方法，并添加`@Scheduled`注解：                 

```java
@Scheduled(fixedRate = 2000)   //单位依然是毫秒，这里是每两秒钟打印一次
public void task(){
    System.out.println("我是定时任务！"+new Date());
}
```

![image-20221125155352390](https://oss.itbaima.cn/internal/markdown/2022/11/25/aGv7f3eBXPsFdYr.png)

我们注意到`@Scheduled`中有很多参数，我们需要指定'cron', 'fixedDelay(String)', or 'fixedRate(String)'的其中一个，否则无法创建定时任务，他们的区别如下：

- fixedDelay：在上一次定时任务执行完之后，间隔多久继续执行。
- fixedRate：无论上一次定时任务有没有执行完成，两次任务之间的时间间隔。
- cron：如果嫌上面两个不够灵活，你还可以使用cron表达式来指定任务计划。

**cron**：https://blog.csdn.net/sunnyzyq/article/details/98597252

## 3.SpringEL表达式

### 3.1外部注入

有些时候，我们甚至可以将一些外部配置文件中的配置进行读取，并完成注入。

我们需要创建以`.properties`结尾的配置文件，这种配置文件格式很简单，类似于Map，需要一个Key和一个Value，中间使用等号进行连接，这里我们在resource目录下创建一个`test.properties`文件：                        

```properties
test.name=只因
```

这样，Key就是`test.name`，Value就是`只因`，我们可以通过一个注解直接读取到外部配置文件中对应的属性值，首先我们需要引入这个配置文件，我们可以在配置类上添加`@PropertySource`注解：

```java
@Configuration
@ComponentScan("com.test.bean")
@PropertySource("classpath:test.properties")   //注意，类路径下的文件名称需要在前面加上classpath:
public class MainConfiguration{
    
}
```

接着，我们就可以开始快乐的使用了，我们可以使用 @Value 注解将外部配置文件中的值注入到任何我们想要的位置，就像我们之前使用@Resource自动注入一样：             

```java
@Component
public class Student {
    @Value("${test.name}")   //这里需要在外层套上 ${ }
    private String name;   //String会被自动赋值为配置文件中对应属性的值

    public void hello(){
        System.out.println("我的名字是："+name);
    }
}
```

`@Value`中的`${...}`表示占位符，它会读取外部配置文件的属性值装配到属性中，如果配置正确没问题的话，这里甚至还会直接显示对应配置项的值：

除了在字段上进行注入之外，我们也可以在需要注入的方法中使用：                    

```java
@Component
public class Student {
    private final String name;

  	//构造方法中的参数除了被自动注入外，我们也可以选择使用@Value进行注入
    public Student(@Value("${test.name}") String name){
        this.name = name;
    }

    public void hello(){
        System.out.println("我的名字是："+name);
    }
}
```

当然，如果我们只是想简单的注入一个常量值，也可以直接填入固定值：                       

```java
private final String name;
public Student(@Value("10") String name){   //只不过，这里都是常量值了，我干嘛不直接写到代码里呢
    this.name = name;
}
```

### 3.2SpEL简单使用

Spring官方为我们提供了一套非常高级SpEL表达式，通过使用表达式，我们可以更加灵活地使用Spring框架。

首先我们来看看如何创建一个SpEL表达式：                 

```java
ExpressionParser parser = new SpelExpressionParser();
Expression exp = parser.parseExpression("'Hello World'");  //使用parseExpression方法来创建一个表达式
System.out.println(exp.getValue());   //表达式最终的运算结果可以通过getValue()获取
```

这里得到的就是一个很简单的 Hello World 字符串，字符串使用单引号囊括，SpEL是具有运算能力的。

我们可以像写Java一样，对这个字符串进行各种操作，比如调用方法之类的：            

```java
Expression exp = parser.parseExpression("'Hello World'.toUpperCase()");   //调用String的toUpperCase方法
System.out.println(exp.getValue());
```

![image-20221125173157008](https://oss.itbaima.cn/internal/markdown/2022/11/25/PZmheYn5EVTvURN.png)

不仅能调用方法、还可以访问属性、使用构造方法等，是不是感觉挺牛的，居然还能这样玩。

对于Getter方法，我们可以像访问属性一样去使用：                        

```java
//比如 String.getBytes() 方法，就是一个Getter，那么可以写成 bytes
Expression exp = parser.parseExpression("'Hello World'.bytes");
System.out.println(exp.getValue());
```

表达式可以不止一级，我们可以多级调用：                  

```java
Expression exp = parser.parseExpression("'Hello World'.bytes.length");   //继续访问数组的length属性
System.out.println(exp.getValue());
```

是不是感觉挺好玩的？我们继续来试试看构造方法，其实就是写Java代码，只是可以写成这种表达式而已：                 

```java
Expression exp = parser.parseExpression("new String('hello world').toUpperCase()");
System.out.println(exp.getValue());
```

![image-20221125173157008](https://oss.itbaima.cn/internal/markdown/2022/11/25/PZmheYn5EVTvURN.png)

它甚至还支持根据特定表达式，从给定对象中获取属性出来：                    

```java
@Component
public class Student {
    private final String name;
    public Student(@Value("${test.name}") String name){
        this.name = name;
    }

    public String getName() {    //比如下面要访问name属性，那么这个属性得可以访问才行，访问权限不够是不行的
        return name;
    }
}            
```

```java
Student student = context.getBean(Student.class);
ExpressionParser parser = new SpelExpressionParser();
Expression exp = parser.parseExpression("name");
System.out.println(exp.getValue(student));    //直接读取对象的name属性
```

拿到对象属性之后，甚至还可以继续去处理：                      

```java
Expression exp = parser.parseExpression("name.bytes.length");   //拿到name之后继续getBytes然后length
```

除了获取，我们也可以调用表达式的setValue方法来设定属性的值：                      

```java
Expression exp = parser.parseExpression("name");
exp.setValue(student, "刻师傅");   //同样的，这个属性得有访问权限且能set才可以，否则会报错
```

除了属性调用，我们也可以使用运算符进行各种高级运算：                     

```java
Expression exp = parser.parseExpression("66 > 77");   //比较运算
System.out.println(exp.getValue());                  
```

```java
Expression exp = parser.parseExpression("99 + 99 * 3");   //算数运算
System.out.println(exp.getValue());
```

对于那些需要导入才能使用的类，我们需要使用一个特殊的语法：              

```java
Expression exp = parser.parseExpression("T(java.lang.Math).random()");   //由T()囊括，包含完整包名+类名
//Expression exp = parser.parseExpression("T(System).nanoTime()");   //默认导入的类可以不加包名
System.out.println(exp.getValue());
```

### 3.3集合相关操作

现在我们的类中存在一些集合类：                  

```java
@Component
public class Student {
    public Map<String, String> map = Map.of("test", "你干嘛");
    public List<String> list = List.of("AAA", "BBB", "CCC");
}
```

我们可以使用SpEL快速取出集合中的元素：

```java
Expression exp = parser.parseExpression("map['test']");  //对于Map这里映射型，可以直接使用map[key]来取出value
System.out.println(exp.getValue(student));                 
```

```java
Expression exp = parser.parseExpression("list[2]");   //对于List、数组这类，可以直接使用[index]
System.out.println(exp.getValue(student));
```

我们也可以快速创建集合：                 

```java
Expression exp = parser.parseExpression("{5, 2, 1, 4, 6, 7, 0, 3, 9, 8}"); //使用{}来快速创建List集合
List value = (List) exp.getValue();
value.forEach(System.out::println);                  
```

```java
Expression exp = parser.parseExpression("{{1, 2}, {3, 4}}");   //它是支持嵌套使用的                   
```

```java
//创建Map也很简单，只需要key:value就可以了，怎么有股JSON味
Expression exp = parser.parseExpression("{name: '小明', info: {address: '北京市朝阳区', tel: 10086}}");
System.out.println(exp.getValue());
```

你以为就这么简单吗，我们还可以直接根据条件获取集合中的元素：             

```java
@Component
public class Student {
    public List<Clazz> list = List.of(new Clazz("高等数学", 4));

    public record Clazz(String name, int score){ }
}
```

```java
//现在我们希望从list中获取那些满足我们条件的元素，并组成一个新的集合，我们可以使用.?运算符
Expression exp = parser.parseExpression("list.?[name == '高等数学']");
System.out.println(exp.getValue(student));                   
```

```java
Expression exp = parser.parseExpression("list.?[score > 3]");   //选择学分大于3分的科目
System.out.println(exp.getValue(student));
```

我们还可以针对某个属性创建对应的投影集合：                  

```java
Expression exp = parser.parseExpression("list.![name]");   //使用.!创建投影集合，这里创建的时课程名称组成的新集合
System.out.println(exp.getValue(student));
```

![image-20221130153142677](https://oss.itbaima.cn/internal/markdown/2022/11/30/yLNHPJnWkoR3Cb2.png)

**安全导航运算符**

安全导航运算符用于避免NullPointerException，它来自Groovy语言。通常，当您有对对象的引用时，您可能需要在访问对象的方法或属性之前验证它是否为空。为了避免这种情况，安全导航运算符返回null而不是抛出异常。

~~~java
Expression exp = parser.parseExpression("name.toUpperCase()");   //如果Student对象中的name属性为null
System.out.println(exp.getValue(student));
~~~

![image-20221130150723018](https://oss.itbaima.cn/internal/markdown/2022/11/30/dojeP5kYcM7KHiv.png)

类似于这种判空问题，我们就可以直接使用安全导航运算符，SpEL也支持这种写法：

```java
Expression exp = parser.parseExpression("name?.toUpperCase()");
System.out.println(exp.getValue(student));
```

当遇到空时，只会得到一个null，而不是直接抛出一个异常：

![image-20221130150654287](https://oss.itbaima.cn/internal/markdown/2022/11/30/tOf3LFsWE4H8BVc.png)

## 4.AOP面向切片

