### 作业要求
#### 作业一：单因子分析
分别计算return_1m、turn_1m、std_1m、std_FF3factor_1m因子并测试效果。

#### 作业二：多因子分析与机器学习模型
参考研报，尝试构造多个价量因子（至少包括5个以上价量因子或原创因子），需给出部分研报中价量因子的描述和分析，以及所有原创因子的描述和分析。最后可通过**线性模型**或**机器学习模型**进行因子结合成最后的alpha，并给出alpha的有效性分析或回测报告。

研究时间区间为2010初-2018年底以内，最少保持两年以上的数据量。
#### 作业三：构建你自己的 GPT 因子工厂
基于**华泰金工研究《GPT因子工厂：多智能体与因子挖掘》** 的方法，模拟构建三个大语言模型智能体：FactorGPT、CodeGPT与 EvalGPT。目标是让这三个“智能体”协同工作，完成从因子生成、代码生成、到因子回测与优化建议输出的闭环。
### 作业提交结构
```
学号+姓名/
├── 作业一/
│    ├── factor_return_1m_raw.csv
│    ├── factor_return_1m_processed.csv
│    ├── factor_return_1m_test.csv
│    ├── factor_turn_1m_raw.csv
│    ├── factor_turn_1m_processed.csv
│    ├── factor_turn_1m_test.csv
│    ├── factor_std_1m_raw.csv
│    ├── factor_std_1m_processed.csv
│    ├── factor_std_1m_test.csv
│    ├── factor_std_FF3_1m_raw.csv
│    ├── factor_std_FF3_1m_processed.csv
│    ├── factor_std_FF3_1m_test.csv
│    ├── report.docx
│    └── code.py/.ipynb
├── 作业二/
│    ├── report.docx
│    └── code.py/.ipynb
└── 作业三/
      └── code.ipynb
```