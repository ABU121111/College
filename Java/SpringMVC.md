# SpringMVC

## 1.MVC理论基础

### 1.1通过XML配置

导入依赖。

~~~xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-webmvc</artifactId>
    <version>6.0.10</version>
</dependency>
~~~

用web.xml中的DispatcherServlet替换tomcat自身的Servlet

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns="https://jakarta.ee/xml/ns/jakartaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="https://jakarta.ee/xml/ns/jakartaee https://jakarta.ee/xml/ns/jakartaee/web-app_5_0.xsd"
         version="5.0">
    <servlet>
        <servlet-name>mvc</servlet-name>
        <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>mvc</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
~~~

为整个web应用配置一个Spring上下文环境（容器），编写一个配置文件Spring.xml

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
        https://www.springframework.org/schema/beans/spring-beans.xsd">
</beans>
~~~

接着需要为DispatcherServlet配置初始化参数来指定刚刚的配置文件

~~~xml
<servlet>
    <servlet-name>mvc</servlet-name>
    <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    <init-param>
      	<!--     指定我们刚刚创建在类路径下的XML配置文件       -->
        <param-name>contextConfigLocation</param-name>
        <param-value>classpath:application.xml</param-value>
    </init-param>
</servlet>
~~~

然后新建一个Controller类进行测试

~~~java
@Controller
public class HelloController {
    @ResponseBody
    @RequestMapping("/")
    public String hello(){
        return "HelloWorld!";
    }
}
~~~

注意需要把这个类注册为Bean，在Spring.xml中加入包扫描

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
        https://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/context https://www.springframework.org/schema/context/spring-context.xsd">
  	<!-- 需要先引入context命名空间，然后直接配置base-package属性就可以了 -->
    <context:component-scan base-package="com.example"/>
</beans>
~~~

### 1.2使用全注解配置

首先配置一个**初始化类**MainInitializer来配置基本的配置类

~~~java
public class MainInitializer extends AbstractAnnotationConfigDispatcherServletInitializer {

    @Override
    protected Class<?>[] getRootConfigClasses() {
        return new Class[]{WebConfiguration.class};   //基本的Spring配置类，一般用于业务层配置
    }

    @Override
    protected Class<?>[] getServletConfigClasses() {
        return new Class[0];  //配置DispatcherServlet的配置类、主要用于Controller等配置，这里为了教学简单，就不分这么详细了，只使用上面的基本配置类
    }

    @Override
    protected String[] getServletMappings() {
        return new String[]{"/"};    //匹配路径，与上面一致
    }
}
~~~

然后创建一个**配置类**，添加一些注解

~~~java
@Configuration
@EnableWebMvc   //快速配置SpringMvc注解，如果不添加此注解会导致后续无法通过实现WebMvcConfigurer接口进行自定义配置
@ComponentScan("com.example.controller")
public class WebConfiguration {
}
~~~

## 2.Contorller控制器

### 2.1配置视图解析器和控制器

导入依赖

~~~xml
<dependency>
    <groupId>org.thymeleaf</groupId>
    <artifactId>thymeleaf-spring6</artifactId>
    <version>3.1.1.RELEASE</version>
</dependency>
~~~

将ViewResolver注册为Bean，在配置类中写：

~~~java
@Configuration
@EnableWebMvc
@ComponentScan("com.example.controller")
public class WebConfiguration {
    //我们需要使用ThymeleafViewResolver作为视图解析器，并解析我们的HTML页面
    @Bean
    public ThymeleafViewResolver thymeleafViewResolver(SpringTemplateEngine springTemplateEngine){
        ThymeleafViewResolver resolver = new ThymeleafViewResolver();
        resolver.setOrder(1);   //可以存在多个视图解析器，并且可以为他们设定解析顺序
        resolver.setCharacterEncoding("UTF-8");   //编码格式是重中之重
        resolver.setTemplateEngine(springTemplateEngine);   //和之前JavaWeb阶段一样，需要使用模板引擎进行解析，所以这里也需要设定一下模板引擎
        return resolver;
    }

    //配置模板解析器
    @Bean
    public SpringResourceTemplateResolver templateResolver(){
        SpringResourceTemplateResolver resolver = new SpringResourceTemplateResolver();
        resolver.setSuffix(".html");   //需要解析的后缀名称
        resolver.setPrefix("classpath:");   //需要解析的HTML页面文件存放的位置，默认是webapp目录下，如果是类路径下需要添加classpath:前缀,默认是resource目录下
        return resolver;
    }

    //配置模板引擎Bean
    @Bean
    public SpringTemplateEngine springTemplateEngine(ITemplateResolver resolver){
        SpringTemplateEngine engine = new SpringTemplateEngine();
        engine.setTemplateResolver(resolver);   //模板解析器，默认即可
        return engine;
    }
}
~~~

接着在根目录下创建html文件

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test</title>
    <script src="static/test.js"></script>
</head>
<body>
    <p>wua</p>
<p th:text="${name}"></p>
</body>
</html>
~~~

使用@Controller注解实现自动装配并且根据@RequestMapping(value = "/")映射HTTP请求到处理方法

~~~java
@Controller   //直接添加注解即可
public class HelloController {

@RequestMapping(value = "/")
public String index(Model model){
    model.addAttribute("name", "Hello World");
    return "index";
}
}
~~~

然后对静态资源进行配置，在配置类中写：

~~~java
    @Override
public void configureDefaultServletHandling(DefaultServletHandlerConfigurer configurer) {
    configurer.enable();   //开启默认的Servlet
}

@Override
public void addResourceHandlers(ResourceHandlerRegistry registry) {
    registry.addResourceHandler("/static/**").addResourceLocations("classpath:/static/");
    //配置静态资源的访问路径
}
~~~

根目录下新建static包test.js文件

~~~js
alert('你好');
~~~

index.html中访问静态资源

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test</title>
    <script src="static/test.js"></script>
</head>
<body>
    <p>wua</p>
<p th:text="${name}"></p>
</body>
</html>
~~~



