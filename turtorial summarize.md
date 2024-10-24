# turtorial 总结

## Week 1 Tutorial 2024

### Q1: STATE and EXPLAIN the THREE ways that an experiment can fail when uncertainties are not considered appropriately.  问题1: 陈述并解释当不确定性没有得到适当考虑时，实验失败的三种方式。

- **<u>LAZINESS</u>: too much work to list the complete set of antecedents or consequents needed to ensure an exceptionless rule and too hard to use such rules.  
  懒惰: 为了确保一个无例外的规则而需要列出完整的前因后果集合的工作量太大，而且很难使用这样的规则。**   
  It can be too cumbersome to list all necessary conditions and rules, making it challenging to account for every scenario.   
  列出所有必要的条件和规则可能过于繁琐，使得对每个场景进行说明变得具有挑战性。  

- **<u>THEORETICAL ignorance</u>: Medical science has no complete theory for the domain.   
  理论上的无知: 医学在这个领域没有完整的理论。** 
  Some fields like medicine, don’t have complete theories for all conditions, meaning the knowledge base is inherently limited.   
  有些领域像医学，没有完整的理论适用于所有的情况，这意味着知识基础是固有的有限的。  

- **<u>PRACTICAL ignorance</u>: Even if we know all the rules, we might be uncertain about a particular patient because not all the necessary tests have been or can be run.   
  实际无知: 即使我们知道所有的规则，我们也可能对某个特定的病人不确定，因为并非所有必要的检查都已经或可以运行。**   
  Even if the rules are known, not every test or condition can be observed or verified in real life, introducing gaps in the system’s ability to make accurate decisions.  
  即使规则是已知的，也不是所有的测试或条件都能在现实生活中被观察或验证，这就给系统做出准确决策的能力带来了空白。  

### Q2: STATE and EXPLAIN the THREE main types of learning in AI. 问题2: 陈述并解释人工智能中的三种主要学习方式。

- The type of learning depends on the feedback to learn from, there are 3 main types of learning, namely **unsupervised learning, supervised learning and reinforcement learning**.  
  学习的类型取决于学习的反馈，有三种主要的学习类型，即非监督式学习、监督式学习和强化学习。
  
  - <u>Unsupervised learning: </u>the agent learns patterns in the input even though no explicit feedback is supplied.  
    非监督式学习: 即使没有提供明确的反馈，代理也会在输入中学习模式。
  
  - <u>Supervised learning: </u>the agent observes some example input–output pairs and learns a function that maps from input to output.   
    监督式学习: 代理观察一些示例输入输出对，并学习一个从输入到输出的映射函数。
  
  - <u>Reinforcement learning: </u>the agent learns from a series of reinforcements—rewards or punishments.    
    强化学习: 代理人从一系列的增援中学习ーー奖励或惩罚。  

### Q3: Discuss the challenges and limitations of current AI technologies.  问题3: 讨论当前人工智能技术的挑战和局限性。

- <u>Data limitations:</u> AI heavily relies on large amounts of high-quality data for training and learning. Limited or biased data can lead to inaccurate or biased AI systems.   
  数据局限性: 人工智能严重依赖大量高质量的数据用于培训和学习。有限或有偏见的数据可能导致不准确或有偏见的人工智能系统。

- <u>Lack of interpretability:</u> Many AI algorithms, such as deep neural networks, are considered black boxes, making it challenging to understand and interpret their decision-making processes.   
  缺乏可解释性: 许多人工智能算法，如深度神经网络，被认为是黑盒子，使其具有挑战性的理解和解释他们的决策过程。

- <u>Computing power and resource requirements:</u> AI algorithms, particularly deep learning, often require significant computing power and resources, which can be a limitation for certain applications or organizations.  
  计算能力和资源需求: 人工智能算法，尤其是深度学习，往往需要大量的计算能力和资源，这可能是某些应用程序或组织的限制。

### Q6: Discuss the importance of understanding prior probabilities when interpreting predictions from AI models.  问题6: 讨论在解释人工智能模型预测时理解先验概率的重要性。

Prior probabilities represent the **baseline likelihood of an event or condition before considering new evidence**.   
先验概率代表在考虑新证据前发生事件或情况的基线可能性。  

In AI predictions, especially in medical or criminal justice applications, understanding these priors is crucial as they influence the model's overall prediction confidence. Ignoring prior probabilities may lead to over-reliance on the model's output without considering the context.   
在人工智能预测中，特别是在医疗或刑事司法应用中，理解这些先验是至关重要的，因为它们影响模型的整体预测信心。忽视先验概率可能导致过度依赖模型的输出而不考虑上下文。

For example, even if a model predicts high probability for a rare disease, the low prior probability of the disease should temper the interpretation. Bayes' theorem exemplifies this by combining prior probabilities with likelihoods to produce more balanced, context-aware predictions.  
例如，即使一个模型预测某种罕见疾病的发生几率很高，疾病的低先验概率也会影响对该模型的解释。贝叶斯定理将先验概率和可能性结合起来，产生更加平衡、上下文感知的预测，就是一个例子。


