[TOC]

[Python 相对导入与绝对导入](http://kuanghy.github.io/2016/07/21/python-import-relative-and-absolute)

[Python导入模块的几种姿势](http://codingpy.com/article/python-import-101/)

## 常见的导入方式

```
import sys     # 导入整个模块
import sys as system  # 导入整个模块并命名
from sys import ...   # 导入sys模块中的子模块
from os import *  # 不推荐，导入该模块下所有的包
```

## `from __future__ import *`

> `__future__` 模块由 [PEP 236](https://link.jianshu.com?t=https://www.python.org/dev/peps/pep-0236/) 提出并加入到 Python 2.1，其存在的主要原因是 Python 的版本升级经常会增加一些新的特性，而 `__future__` 模块将一些新版本中将会增加的新的特性进行声明，同时使得旧版本可以使用这些新的语法特性。

这也就是说，如果你要想在低版本中使用高版本的特性，那么`from __future__ import ...`可以很好的帮你实现，这也意味着如果你在你的代码中使用`from __future__ import ...`， 会提高你代码的向下兼容性。

需要注意的是：

> - 如果你用的是 Python 2.1 以前的版本，是没办法使用 `__future__` 的。
> - `__future__` 模块的导入一定要放在最上方，也就是在所有其它模块之前导入。

## `from __future__ import absolute_import`

> 这句话的意思是将所有导入视为绝对导入，指的是禁用`implicit relative import`（隐式相对导入）, 但并不会禁掉 `explicit relative import`（显示相对导入）。

## python 库搜索路径

当你导入时，会按照以下路径按顺序来搜索你要导入的文件，python的搜索路径构成了`sys.path`。：

1. 在当前目录下搜索该模块
2. 在环境变量 PYTHONPATH 中指定的路径列表中依次搜索
3. 在 Python 安装路径的 lib 库中搜索
4. 也许会用到`.pth` 文件，但一般不用

python 所有加载的模块信息都存放在 `sys.modules` 结构中，当 import 一个模块时，会按如下步骤来进行

- 如果是 `import A`，检查 sys.modules 中是否已经有 A，如果有则不加载，如果没有则为 A 创建 module 对象，并加载 A
- 如果是 `from A import B`，先为 A 创建 module 对象，再解析A，从中寻找B并填充到 A 的 `__dict__`中

## 包内导入

包内导入就是**包内的模块导入包内部的模块**。举个例子：

```
example
   --- main
       --- test.py
   --- model
       --- view.py
```

这是常见的一个结构，将项目的不同模块区分开，那么此时的包内导入如何就是在`test.py` 中导入`view.py` 。

## 绝对导入与相对导入

首先，注意一点，绝对导入与相对导入是针对**包内导入**而言的。其中：

> - 绝对导入的格式为： `import A.B` 或`from A import B`
> - 相对导入的格式为：`from .. import B`, `.`代表当前模块，`..`代表上层模块，`...`代表上上层模块，依次类推。

