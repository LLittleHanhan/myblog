1. 构造函数为什么不能是虚函数
在派生类中，基类的构造函数会被自动调用，用于初始化基类的成员。因此，构造函数没有被覆盖的必要
1. 为什么父类的析构函数需要是虚函数
https://csguide.cn/cpp/object_oriented/why_destructor_be_virtual.html