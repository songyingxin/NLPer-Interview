## 为什么说python线程是伪线程？

但是在python中，**python虚拟机要求在主循环中同时只能有一个控制线程在运行，这也就意味着即使python解释器中可以运行多个线程，但是在任意时刻只有一个线程会被python解释器执行。**

而这正是由**GIL（全局解释器锁）**来控制的，它保证了同一时刻只能有一个线程运行，而在python多线程环境下，python虚拟机按照下面的 方式运行：

1. 设置GIL

2. 切换进一个线程取运行

3. 执行下面操作之一：

   > - 执行指定数量的字节码指令
   > - 线程主动让出控制权（time.sleep())

4. 把线程设置会睡眠状态（切换出线程）

5. 解锁GIL

6. 重复以上步骤

这也就是为什么说python的多线程适合于IO密集型，而不适合计算密集型任务。

## 什么是守护线程

守护线程可以视为其余非守护线程的保姆，只有所有非守护线程都退出了，守护线程才会终止。

threading模块支持守护线程，其工作方式是:守护线程一般是一个等待客户端请求服务的服务器。如果没有客户端请求,守护线程就是空闲的。

## python中的多线程

python中提供了很方便的库来提供多线程，该库就是Threading库。

## Threading.Thread类

首先我们先来介绍一下它的初始化函数，再来介绍它的相关属性，最后，我们介绍它常用的一些方法。

###`__init__`

```
class threading.Thread(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
```

> - group: 预留参数，用于扩展
> - target： run()方法所调用的函数，默认为None。 
> - name： 线程名字，默认为" Thread-*N*" 
> - args： 调用target时的参数列表
> - kwargs： 调用target时的关键字列表
> - daemon: 为True，表示启动后台线程（对于需要长时间运行的线程或者需要一直运行的后台任务，你应该考虑使用后台线程）

### Thread类的属性

### 1. name

> 获取和设置线程的名字，可更改。

### 2. ident

> 获取线程的标识符。线程标识符是一个非零整数，只有在调用了start()方法之后该属性才有效，否则它只返回None。

### 3. daemon

> 一个 boolean 值表示该进程是不是后台进程(守护进程）。True： 后台进程   Flase： 非后台线程，可更改。

### Thread类方法

### 1. start()

> 启动线程活动

每个线程对象必须调用最多一次start()函数，

### 2. run()

定义线程功能的方法（通常在子类中被应用开发者重写

### 3. join()

```
join(timeout=None)
```

设置主线程是否同步阻塞自己来待此线程执行完毕。如果不设置的话则主进程会继续执行自己的，在结束时根据 setDaemon 有无注册为守护模式的子进程，有的话将其回收，没有的话就结束自己，某些子线程可以仍在执行

 主线程启动若干个子线程后，可以继续执行主线程的代码，也可以等待所有的子线程执行完毕后继续执行主线程，这里需要用到的就是 join 方法，子线程通过调用 join 可以告诉主线程，你必须等着我，我完事了你才可以再往下执行。

### 4. is_alive()

> 判断线程是否alive
>
> True: alive      False： not alive

## 创建线程的几种方法

### 1. 第一种： 派生Thread的子类，并创建子类的实例（推荐）

> - 第一步：创建一个线程子类，该类继承`threading.Thread`类
> - 第二步：复写该子类中的run方法
> - 第三步：写该子线程要执行的功能函数模块
> - 第四步：创建Thread子类实例，运行该线程

```python
class ExampleThread(threading.Thread):

    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func   # 传入的函数（python中允许向函数中传递函数）
        self.args = args   # 表示要传递到函数的参数信息

    def run(self):
        self.func(*self.args)

def example(参数列表):
    # 子线程要执行的功能实现

if __name__ == "__main":
    t = ExampleThread(example, 参数列表, example.__name__)
    t.start()
```

###第二种： 创建Thread实例，传递给它一个函数 

该方法是最简单的方法，但是不推荐你使用，因为其不符合面向对象的思想。

> - 第一步：创建一个子线程要执行的功能函数模块
> - 第二步：创建Thread实例，运行该线程

我们看到，该方法与上面的方法相比，更为简洁，这两种方法选一个就好，我个人偏向第一种。

```python
def example(参数列表):
    # 子线程要执行的功能实现

if __name__ == "__main":
    t = threading.Thread(target=example, args=参数列表)
    t.start()
```







