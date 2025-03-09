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

### 2.2请求映射

在Controller类中添加@RequestMapper注解，从而能够对地址进行映射

Ps:地址与类只能多对一，不能一对多

~~~java
@RequestMapping({"/index", "/test"})
public ModelAndView index(){
    return new ModelAndView("index");
}
~~~

当然可以使用衍生注解直接设定为指定类型的请求映射，比如POST，GET等

~~~java
@PostMapping(value = "/index")
public ModelAndView index(){
    return new ModelAndView("index");
}
~~~

进一步，我们可以使用params属性来指定请求必须带有哪些参数

~~~java
@RequestMapping(value = "/index", params = {"username", "password"})
public ModelAndView index(){
    return new ModelAndView("index");
}
~~~

Plus玩法，可以加表达式

~~~java
//不允许带某个参数
@RequestMapping(value = "/index", params = {"!username", "password"})
public ModelAndView index(){
    return new ModelAndView("index");
}
//不等于某个参数 指定参数的值
@RequestMapping(value = "/index", params = {"username!=test", "password=123"})
public ModelAndView index(){
    return new ModelAndView("index");
}
~~~

header属性与params一致，但是要求请求中必须携带某个属性，比如：

~~~java
@RequestMapping(value = "/index", headers = "!Connection")
public ModelAndView index(){
    return new ModelAndView("index");
}
~~~

那么，如果请求头中携带了`Connection`属性，将无法访问。其他两个属性：

- consumes： 指定处理请求的提交内容类型（Content-Type），例如application/json, text/html;
- produces: 指定返回的内容类型，仅当request请求头中的(Accept)类型中包含该指定类型才返回；

### 2.3请求参数获取

我们只需要为方法添加一个形式参数，并在形式参数前面添加`@RequestParam`注解即可：   

```java
@RequestMapping(value = "/index")
public ModelAndView index(@RequestParam("username") String username){
    System.out.println("接受到请求参数："+username);
    return new ModelAndView("index");
}
```

我们需要在`@RequestParam`中填写参数名称，参数的值会自动传递给形式参数，我们可以直接在方法中使用，注意，如果参数名称与形式参数名称相同，即使不添加`@RequestParam`也能获取到参数值。

还可以设置一个默认值，当参数缺失时可以直接使用默认值

~~~java
@RequestMapping(value = "/index")
public ModelAndView index(@RequestParam(value = "username", required = false, defaultValue = "伞兵一号") String username){
    System.out.println("接受到请求参数："+username);
    return new ModelAndView("index");
}
~~~

还可以直接将请求参数传递给一个实体类：

~~~java
@Data
public class User {
    String username;
    String password;
}
~~~

注意必须携带**set方法**或者**构造方法**中所包含的全部参数，请求参数会自动匹配

~~~java
@RequestMapping(value = "/index")
public ModelAndView index(User user){
    System.out.println("获取到cookie值为："+user);
    return new ModelAndView("index");
}
~~~

通过使用`@CookieValue`注解，我们也可以快速获取请求携带的Cookie信息：                   

```java
@RequestMapping(value = "/index")
public ModelAndView index(HttpServletResponse response,
                          @CookieValue(value = "test", required = false) String test){
    System.out.println("获取到cookie值为："+test);
    response.addCookie(new Cookie("test", "lbwnb"));
    return new ModelAndView("index");
}
```

同样的，Session也能使用注解快速获取：          

```java
@RequestMapping(value = "/index")
public ModelAndView index(@SessionAttribute(value = "test", required = false) String test,
                          HttpSession session){
    session.setAttribute("test", "xxxx");
    System.out.println(test);
    return new ModelAndView("index");
}
```

### 2.4重定向与请求转发

**重定向**

~~~java
@RequestMapping("/index")
public String index(){
    //跳转到home
    return "redirect:home";
}

@RequestMapping("/home")
public String home(){
    return "home";
}
~~~

**请求转发**

~~~java
@RequestMapping("/index")
public String index(){
    return "forward:home";
}

@RequestMapping("/home")
public String home(){
    return "home";
}
~~~

### 2.5Bean的Web作用域

- request：对于每次HTTP请求，使用request作用域定义的Bean都将产生一个新实例，请求结束后Bean也消失。
- session：对于每一个会话，使用session作用域定义的Bean都将产生一个新实例，会话过期后Bean也消失。

对于一个实体类注册为Bean并限定其作用域

~~~java
@Component
@SessionScope
public class User {
    String name;
    String password;
}
~~~

然后在配置文件中添加包扫描

~~~java
@Configuration
@EnableWebMvc
@ComponentScans({
        @ComponentScan("com.example.controller"),
        @ComponentScan("com.example.entity")
})
~~~

接着重写方法，自动装配

~~~java
@Autowired
User user;
@RequestMapping(value = "/test")
//返回的内容会直接输出
@ResponseBody
public String index(){
    return user.toString();
}
~~~

## 3.RestFul

通过拼接url来将参数传到服务端

~~~xml
http://localhost:8080/mvc/index/123456
~~~

~~~java
@RequestMapping("/index/{str}")
//映射路径并使用
public String index(@PathVariable String str) {
    System.out.println(str);
    return "index";
}
~~~

## 4.Interceptor

拦截器类似过滤器，都是用来**拦截非法请求**的，但是过滤器是**在Servelt之前**，经过过滤后才会到达Servelt；而拦截器是位与**Servelt与RequestMappering**之间，相当于DispatcherServelt将请求先交给拦截器再交给对应的Controller方法

### 4.1创建拦截器

首先需要实现一个接口

~~~java
public class MainInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        System.out.println("我是处理之前！");
        return true;   //只有返回true才会继续，否则直接结束
    }

    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {
        System.out.println("我是处理之后！");
    }

    @Override
    //类似Finally，无论有没有异常都会执行
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {
      	//在DispatcherServlet完全处理完请求后被调用
        System.out.println("我是完成之后！");
    }
}
~~~

接着在注册类配置

~~~java
@Override
public void addInterceptors(InterceptorRegistry registry) {
    registry.addInterceptor(new MainInterceptor())
      .addPathPatterns("/**")    //添加拦截器的匹配路径，只要匹配一律拦截
      .excludePathPatterns("/home");   //拦截器不进行拦截的路径
}
~~~

### 4.2多级拦截器

注册二号拦截器

~~~java
@Override
public void addInterceptors(InterceptorRegistry registry) {
  	//一号拦截器
    registry.addInterceptor(new MainInterceptor()).addPathPatterns("/**").excludePathPatterns("/home");
  	//二号拦截器
    registry.addInterceptor(new SubInterceptor()).addPathPatterns("/**");
}
~~~

结果如下

~~~properties
一号拦截器：我是处理之前！
二号拦截器：我是处理之前！
我是处理！
二号拦截器：我是处理之后！
一号拦截器：我是处理之后！
二号拦截器：我是完成之后！
一号拦截器：我是完成之后！
~~~

和多级Filter相同，在处理之前，是按照顺序从前向后进行拦截的，但是处理完成之后，就按照倒序执行处理后方法，而完成后是在所有的`postHandle`执行之后再同样的以倒序方式执行。

与单个拦截器的情况一样，一旦拦截器返回false，那么之后无论有无拦截器，都**不再继续**。

## 5.异常处理

自定义一个异常控制器，一旦出现异常会转接到此控制器执行：

~~~java
@ControllerAdvice
public class ErrorController {

    @ExceptionHandler(Exception.class)
    public String error(Exception e, Model model){  //可以直接添加形参来获取异常
        e.printStackTrace();
        model.addAttribute("e", e);
        return "500";
    }
}
~~~

编写异常页面

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
  500 - 服务器出现了一个内部错误QAQ
  <div th:text="${e}"></div>
</body>
</html>
~~~

修改index方法，抛出异常

~~~java
@RequestMapping("/index")
public String index(){
    System.out.println("我是处理！");
    if(true) throw new RuntimeException("您的氪金力度不足，无法访问！");
    return "index";
}
~~~

## 6.JSON数据格式

实现前后端分离需要有一种高效的数据传送方式，JSON横空出世。

如何快速创建JSON格式数据，首先需要导入依赖

~~~xml
<dependency>
      <groupId>com.alibaba.fastjson2</groupId>
      <artifactId>fastjson2</artifactId>
      <version>2.0.34</version>
</dependency>
~~~

存放数据

~~~java
@RequestMapping(value = "/index")
public String index(){
    JSONObject object = new JSONObject();
    object.put("name", "杰哥");
    object.put("age", 18);
    System.out.println(object.toJSONString());   //以JSON格式输出JSONObject字符串
    return "index";
}
~~~

如果是数组那么就使用JSONArray

~~~java
@RequestMapping(value = "/index")
public String index(){
    JSONObject object = new JSONObject();
    object.put("name", "杰哥");
    object.put("age", 18);
    JSONArray array = new JSONArray();
    array.add(object);
    System.out.println(array.toJSONString());
    return "index";
}
~~~

还可以创建实体类并将其转换为JSON格式数据，注意要先配置转换器

~~~xml
<dependency>
    <groupId>com.alibaba.fastjson2</groupId>
    <artifactId>fastjson2-extension-spring6</artifactId>
    <version>2.0.34</version>
</dependency>
~~~

然后在配置文件中写：

~~~java
@Override
public void configureMessageConverters(List<HttpMessageConverter<?>> converters) {
    converters.add(new FastJsonHttpMessageConverter());
}
~~~

然后我就得逞了

~~~java
@RequestMapping(value = "/data", produces = "application/json")
@ResponseBody
public Student data(){
    Student student = new Student();
    student.setName("杰哥");
    student.setAge(18);
    return student;
}
~~~

## 7.文件上传下载

利用SpringMVC，我们可以很轻松地实现文件上传和下载，我们需要在MainInitializer中添加一个新的方法：                   

```java
public class MainInitializer extends AbstractAnnotationConfigDispatcherServletInitializer {

    ...

    @Override
    protected void customizeRegistration(ServletRegistration.Dynamic registration) {
      	// 直接通过registration配置Multipart相关配置，必须配置临时上传路径，建议选择方便打开的
        // 同样可以设置其他属性：maxFileSize, maxRequestSize, fileSizeThreshold
        registration.setMultipartConfig(new MultipartConfigElement("/Users/nagocoler/Download"));
    }
}
```

接着我们直接编写Controller即可：

```java
@RequestMapping(value = "/upload", method = RequestMethod.POST)
@ResponseBody
public String upload(@RequestParam MultipartFile file) throws IOException {
    File fileObj = new File("test.png");
    file.transferTo(fileObj);
    System.out.println("用户上传的文件已保存到："+fileObj.getAbsolutePath());
    return "文件上传成功！";
}
```

最后在前端添加一个文件的上传点：                    

```html
<div>
    <form action="upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
</div>
```

这样，点击提交之后，文件就会上传到服务器了。

下载其实和我们之前的写法大致一样，直接使用HttpServletResponse，并向输出流中传输数据即可。             

```java
@RequestMapping(value = "/download", method = RequestMethod.GET)
@ResponseBody
public void download(HttpServletResponse response){
    response.setContentType("multipart/form-data");
    try(OutputStream stream = response.getOutputStream();
        InputStream inputStream = new FileInputStream("test.png")){
        IOUtils.copy(inputStream, stream);
    }catch (IOException e){
        e.printStackTrace();
    }
}
```

在前端页面中添加一个下载点：                  

```html
<a href="download" download="test.png">下载最新资源</a>
```



