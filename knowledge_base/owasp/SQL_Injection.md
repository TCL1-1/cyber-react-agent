# OWASP Top10 SQL Injection


## 类型

Injection


## 原理

SQL Injection 是由于应用程序没有正确过滤用户输入，
导致攻击者可以修改SQL语句。


例如：

正常:

SELECT * FROM users WHERE id=1


攻击:

SELECT * FROM users WHERE id=1 OR 1=1



## 危害

- 数据泄露
- 数据修改
- 绕过认证
- 获取服务器权限


## 防御方法


1. 使用参数化查询

例如:

PreparedStatement


2. 输入验证


3. 最小数据库权限


4. 使用ORM框架

