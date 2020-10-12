<font color='red'>Pytorch版本：1.6.0</font>

<font color='red'>上版更新：分布式训练，DDP(DistributedDataParallel)</font>

<font color='red'>当前更新：nn.SyncBatchNorm，dataSampler（更新中）</font>

下载链接 https://github.com/liuzhaoo/Pytorch-API-and-Tutorials-CN/releases/tag/v0.01

也可直接下载上方pdf文件

:new:  新增了markown文件，但由于主题原因，建议下载pdf使用





<font color='orang'>本文档是对Pytorch官方文档和教材学习及查阅过程中的笔记，不仅对原文档做了便于理解的翻译，还对一些重要的部分进行了解释说明。</font>



<font  color='orang' >相比官方文档，本文以最简洁的方式呈现了Pytorch一些功能和API的使用方法，对于这些方法实现的细节和原理不做过多的介绍，而是给出了原文链接，需要了解的读者可自行查看；同时，本文保留了文档中的代码示例，对复杂脚本进行了代码分析，希望可以帮助读者更快地理解相关内容，节省读者的时间；最后，本项目尚未包含官方文档所有内容，仍在持续更新中。</font>



***NOTE***

- <font color='orang' >本人在查看官方文档时，有时看懂一个功能需要很长时间。所以每次都会记录下来，也因此萌生了将其总结到一起，编写一个文档的想法。既是自己的学习历程，也希望能帮到更多的初学者。</font>
- <font color='orang' >本文档与官方文档结构类似，对每个函数（API）都有书签直接定位，可以作为学习资料，也可作为速查手册。</font>
- <font color='orang' >文中添加了很多链接，以紫色字体显示，便于读者快速转到对应的官方页面，进行进一步的了解。</font>
- <font color='orang' >文档仍在持续更新中，由于是作者一个人在编写，又限于本人课题压力，无法做到定时更新。</font>









<font color='orange'>最后，如果你想对本项目做出贡献，为学习者提供便利，欢迎联系我！</font>

415091lz@gmail.com