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

​              java              复制代码                          

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

要注册这个类的Bean，只需要添加@Component即可，然后配置一下包扫描：                   

```java
@Configuration
@ComponentScan("com.test.bean")   //包扫描，这样Spring就会去扫描对应包下所有的类
public class MainConfiguration {

}
```

