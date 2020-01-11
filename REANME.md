# 烟雾识别

## 1.目标
识别监控拍下的图片中，是否有烟雾

## 2.todo-list
- [x] 训练数据与代码分离
- [x] config.py 实现在服务器（Linux），本机（Windows）的不同配置
- [x] 使用DataGenerator
- [ ] 使用faster-rncc
- [ ] 为train添加命令行参数，对应不同参数进行训练
- [ ] 重构output文件夹，对每一组参数，新建一个文件夹，储存trainlog

## 3.部分文件功能
- run_prj.sh

    在服务器上启动项目

