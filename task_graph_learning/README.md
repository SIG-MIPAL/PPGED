### 参考该篇论文构建DO的概率转移图
https://github.com/fpv-iplab/Differentiable-Task-Graph-Learning

### 构建动态链接库

---
clone下来的Differentiable-Task-Graph-Learning-main是任务图相关的代码，想要新增任务图模型，需要在Differentiable-Task-Graph-Learning-main/lib/src/taskgraph/task_graph_learning/_models.py中新增模型，并重新在Differentiable-Task-Graph-Learning-main文件夹下执行`python -m pip install -e ./lib`

