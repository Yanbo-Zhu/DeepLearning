
RL： Reinforcement Learning
 Aim of RL: Learn to make sequence of good decisions, i.e. that contribute to an overall goal.

![](image/Pasted%20image%2020260206234023.png)


Definition: Reinforcement learning is a type of machine learning where
 an agent learns to make decisions
 by interacting with an environment (evaluation of the current situation).
 The goal is to maximize cumulative rewards (positive feedback) over time  by taking actions that lead to desirable outcomes.


 RL does not require labeled training data
 Instead, RL relies on trial-and-error experiences (exploration) and reward feedback

![](image/Pasted%20image%2020260206234442.png)

# 1 Markov Decision Process (MDP)

 Mathematically, RL is often formalized as a Markov Decision Process, which consists of:

 Goal: find an optimal strategy that maximizes the expected cumulative reward ("return")

![](image/Pasted%20image%2020260206234517.png)


Definition: The Markov property states that the future evolution of a process depends only on its
present state, not on how it arrived there.

State sufficiency: once the current state is known, all past information becomes irrelevant for
predicting the future.


Example: navigating an unknown city with the goal of arriving at a specific target address 
 at each intersection choose a direction
 move to the next intersection

The next location depends only on the current location, not on the full path taken to arrive there.
![](image/Pasted%20image%2020260206234610.png)



## 1.1 MDP Example



 Current state 𝑠 ∈ 𝑆: (next state 𝑠􏰇)
     Own (ego) position and speed (e.g., lane and velocity)
     Position and speed of other cars (e.g., car ahead, cars in adjacent lanes) 
     Distance to the car in front
     Maximum ego acceleration
     Traffic rules (e.g., speed limits, no-passing zones)

 Terminal states:
     Success: ego returned to lane ahead of the passed vehicle with safe gap 
     Failure: collision or forced hard braking below safety threshold
     Abort: decide to give up and remain behind

![](image/Pasted%20image%2020260206234914.png)

---

 Overtaking in a car modeled as Markov Decision Process (𝑆, 𝐴, 𝑃, 𝑅, 𝛾):
     Current state 𝑠

 Current action 𝑎:
 Maintain speed (stay behind)
 Accelerate
 Decelerate
 Change lane left
 Change lane right
 Return to the original lane after overtaking


---

 Overtaking in a car modeled as Markov Decision Process (𝑆, 𝐴, 𝑃, 𝑅, 𝛾): 
 Current state 𝑠
 Current action 𝑎

 Transition probabilities 𝑃(𝑠′ |𝑎, 𝑠): determine how the system moves from one
state 𝑠 to the next 𝑠′ based on actions 𝑎 and environmental factors.
 If the gap is large and no car on adjacent lane: overtaking state has high probabilities
 If another car appears in the adjacent lane, aborting the maneuver has high probabilities

![](image/Pasted%20image%2020260206235219.png)


---

 Overtaking in a car modeled as Markov Decision Process (𝑆, 𝐴, 𝑃, 𝑅, 𝛾):  Current state 𝑠
 Current action 𝑎
 Transition probabilities 𝑃(𝑠′ |𝑎, 𝑠)

 Rewards 𝑅(𝑠, 𝑎, 𝑠􏰇): based on safety and the efficiency/success of overtaking: 
 +10 for successful overtaking within a reasonable time
 -40 for an unsafe action (e.g., cutting off another car, illegal lane change) 
 -5 for staying behind too long
 -100 for a collision
 +2 for maintaining a safe distance

![](image/Pasted%20image%2020260206235126.png)

---

 Overtaking in a car modeled as Markov Decision Process (𝑆, 𝐴, 𝑃, 𝑅, 𝛾): 
 Current state 𝑠
 Current action 𝑎
 Transition probabilities 𝑃(𝑠′ |𝑎, 𝑠) 
 Rewards 𝑅(𝑠, 𝑎, 𝑠􏰇)

 Policy 𝜋 (𝑎 𝑠): maps states to actions. The policy might learn to: 
 Accelerate if the road is clear.
 Wait if the adjacent lane is occupied.
 Change lanes if it's safe.
 Decelerate if another car is ahead in the new lane.

![](image/Pasted%20image%2020260206235205.png)


# 2 Applications

 Robotics: RL is used to train robots for locomotion, object manipulation, and industrial automation (e.g., robotic arms learning to assemble parts).
 Autonomous systems: RL helps self-driving cars learn lane-following, obstacle avoidance, and adaptive cruise control through simulation-based learning.
 Drones and UAVs: RL is applied for trajectory optimization, collision avoidance, and efficient energy management.
 Design Optimization: RL assists in topology optimization of mechanical structures to enhance strength while minimizing weight.
 Predictive Maintenance: RL-based scheduling of maintenance operations reduces downtime in mechanical and industrial systems.

Application areas
 Robotics: navigation, manipulation, and grasping, human interaction 
 Control systems: design and optimize control systems
 Resource allocation and optimization: HPC job scheduling
 Autonomous systems: drones, cars, ...
 ... many more

Robots are capable of doing all ranges of things and motions (i.e. if you use a remote control, robots can do quite everything) ->  building truly helpful/useful robots is a software challenge, not a hardware challenge.

# 3 Agent

The agent is the decision-maker in RL. It observes the environment, selects actions based on a policy, and learns from the rewards it receives. The agent's goal is to maximize long-term cumulative rewards by improving its decision-making strategy.

 Example:
 A self-driving car deciding when to change lanes. 
 A robotic arm learning to grasp objects.


![](image/Pasted%20image%2020260206235657.png)



# 4 State 


 A state 𝑆 represents the current situation of the agent within the environment. It encodes all the
necessary information the agent needs to make a decision.
     The state can be fully observable (i.e., contains all relevant information) or
     partially observable (some information is hidden, agent works with observations instead of full states).

![](image/Pasted%20image%2020260206235908.png)


# 5 Action 
 An action 𝐴 is a decision made by the agent that affects the environment. The set of all possible
actions is called the action space.
     Discrete actions: A limited set of choices. E.g. {accelerate, decelerate, change lane} 
     Continuous actions: Any value within a range. E.g. steering angle `[−30°, 30°]`

 Example:
 For a robotic arm, actions could be moving a joint by a certain angle.
 For a game-playing agent, actions might be pressing buttons on a controller.

![](image/Pasted%20image%2020260206235945.png)

# 6 Reward 

 The reward 𝑅 is the numerical feedback signal that tells the agent how good or bad an action
was. The agent's objective is to maximize the total accumulated reward.
     Immediate reward: Given after each action.
     Delayed reward: Sometimes actions lead to future rewards, requiring the agent to plan ahead.


 Example:
     +10 for successful overtaking 
     -100 for unsafe maneuver


![](image/Pasted%20image%2020260207000024.png)

agent 根据 s_t 得到 该做什么 A_t
S_t 状态下 做了 A_t 得到 s_t+1

根据 S_t+1, S_t 和 A_t   得到  r_t, 就是说  tells the agent how good or bad an action
was. 

## 6.1 Reward 的来源与传递方式

Reward 的获取方式主要取决于环境与 Agent 的交互模式 

1  环境直接返回（最常见）
Agent 执行动作 a_t 后，环境转移到新状态 s_{t+1}，并直接返回一个奖励值 r_t。
observation, reward, done, info = env.step(action)

Agent 在每一步的 reward 会自动从环境获得。


---

2 Agent 内部计算
当奖励规则复杂或依赖于 Agent 内部状态时，可在 Agent 内部计算。

示例：自动驾驶 Agent 的奖励可能综合了速度、舒适度、安全度等多个子项。

```
class MyAgent:
    def compute_reward(self, state, action):
        speed_reward = state.speed * 0.1
        safety_penalty = -abs(state.lane_offset) * 10
        return speed_reward + safety_penalty
    
    def step(self, state):
        action = self.policy(state)
        reward = self.compute_reward(state, action)  # 内部计算奖励
        self.learn(reward)
```


3  从信息流中解析
环境可能将奖励包含在 info 字典中，需要 Agent 主动提取。

示例：某些环境将原始奖励和整形奖励都放在 info 里。
```
observation, _, done, info = env.step(action)
reward = info['original_reward']  # 或 info['shaped_reward']
```


## 6.2 关键问题与解决方案
问题1：奖励延迟
场景：奖励在动作执行多步后才出现（如围棋最终胜负）。

解决方案：

信用分配：使用 TD-Learning（如 Q-Learning）将最终奖励反向传播到关键步骤。

奖励整形：设计中间奖励引导 Agent（如距离目标越近给予小奖励）。

问题2：奖励稀疏
场景：大多数步骤 reward=0，只有少数步骤有正/负奖励（如机器人找到目标）。

解决方案：

内在好奇心模块：给探索新状态附加内在奖励。

分层强化学习：将大任务分解为子任务，每个子任务设置子奖励。

问题3：多目标奖励
场景：需要平衡多个竞争性目标（如速度 vs 安全）。

解决方案：

加权求和：总奖励 = w1 * 奖励1 + w2 * 奖励2

多目标 RL：使用帕累托最优等方法来处理向量化奖励。



# 7 Policy

 A policy 𝜋 defines the agent's strategy for choosing actions based on the current state. It can be: 
     Deterministic: Always chooses the same action for a given state 𝑎 = 𝜋 𝑠
     Stochastic: Chooses actions probabilistically 𝜋(𝑎 ∣ 𝑠) i.e. the probability of action 𝑎 in state 𝑠.
 The policy an be:
     Fixed (rule-based, tabular): Manually designed strategies.
     Learned (adaptive): Improves over time using RL techniques.

 Example:
     A robotic arm's policy might prioritize grasping the object from an angle that minimizes slippage.

![](image/Pasted%20image%2020260207000257.png)

# 8 Reinforcement Learning: How to learn


 How does the learning come into play? When state or action space is too large for tabular methods
![](image/Pasted%20image%2020260207000842.png)

 The agent as a deep learning model: in modern RL, the agent is often a deep neural network. 
     Inputs: state 𝑠 (the agent observes the states of the environment)
     Outputs: action 𝑎 (agent selects an action)
     Training aim: pick the best actions to maximize long-term rewards

 Specifically: neural networks are used as function approximators to represent mappings for
![](image/Pasted%20image%2020260207001438.png)
directly from high dimensional inputs such as images, sensor streams, or large feature vectors.

enables generalization across similar states
 makes reinforcement learning feasible in complex domains like raw pixel control, robotics

## 8.1 Learning Strategies
There is no ground truth data - even if there was, RL would only replicate it
High-level procedure:
1. The agent interacts with the environment using its current policy. 
2. It observes rewards and next states.
3. It updates its policy to increase expected future return.
4. It repeats.

 There is no fixed dataset. The data distribution depends on the current policy, so learning and data collection are coupled.
 There is a need for a safe (!) environment to run the trial-and-error training

---

Exploration vs. Exploitation 
Fundamental tension in RL training

Exploitation
 means choosing the action that currently looks best according to the learned policy.
 It maximizes immediate expected reward given present knowledge.
 Agent that only exploits:
     can converge prematurely to a suboptimal policy because it never discovers better alternatives.

Exploration
means deliberately trying actions that are uncertain or currently estimated as suboptimal
 It gathers information that may improve future decisions.
 Agent that only explores:
     fails to accumulate reward efficiently (will never learn from previous experiences)

Approaches: Stochastic policies (sample from a probability distribution over actions); Epsilon greedy; ...

---

 Approaches organized around what is being learned.

 Value based methods learn a value function and derive the policy indirectly. Core objects 
     State value 𝑉 𝑠
     Action value 𝑄(𝑠, 𝑎)
     Policy: e.g. choosing the action with maximal action value
     Methods: Bellman equation, Q-Learning, Deep Q Networks
sample efficient in discrete action spaces

 Policy based methods learn the policy directly
     Core object is a learnable policy 𝜋􏰀 𝑎 𝑠)
     Training optimizes expected return a performance objective 
     Methods: Policy gradient learning

naturally handles continuous or high dimensional action spaces

![](image/Pasted%20image%2020260207002141.png)


## 8.2 Value Function


 The value function 𝑉(𝑠) estimates the expected cumulative reward an agent can obtain from a given state 𝑠, assuming it follows a certain policy 𝜋.

 Intuition 𝑉􏰏(𝑠)
     If state 𝑠 and policy 𝜋 lead to high rewards in the future, 𝑉􏰏(𝑠) has a high value.
     If state 𝑠 and policy 𝜋 lead to failure or low rewards, 𝑉􏰏(𝑠) has a low value.

 Example: in a chess game, a board position that often leads to a win (assuming that all players stick to their game plan) has a high value.

![](image/Pasted%20image%2020260207002312.png)


## 8.3 Q-Function (Action-Value Function)

The Q-function 𝑄(𝑠, 𝑎) gives the expected cumulative reward for taking action 𝑎 in state 𝑠 and following policy 𝜋 afterwards.

Q(S，a) measures how good it is to choose action 𝑎 in state 𝑠, taking into account all future condequences under the policy 

 Optimal Q-function 𝑄∗(𝑠, 𝑎): 𝑄∗ 𝑠, 𝑎 = max 𝑄􏰏(𝑠, 𝑎)

 Value function 𝑽(𝒔) vs. Q-function 𝑸(𝒔, 𝒂):
 𝑽(𝒔) tells the value of being in a state
 𝑸(𝒔, 𝒂) tells the value of taking a specific action in that state

 Discount factor 𝛾: controls future rewards compared to immediate rewards (default: 𝛾 = 0.9) 
 𝛾 → 0: agent only considers immediate rewards.
 𝛾 → 1 : agent values long-term rewards equally to immediate rewards.

![](image/Pasted%20image%2020260207002448.png)

## 8.4 Deep Q-Learning: Training Process


1 Initialize the neural network with random weights.
2 Playepisodesintheenvironment:
(1) Observe state 𝑆􏰅.
(2) Pick action 𝐴􏰅 (exploration vs. exploitation).
(3) Get reward 𝑅􏰅􏰆􏰂 and next state 𝑆􏰅􏰆􏰂.
(4) Store the experience in memory (Replay Buffer)

3 Train the neural network:
(1) Sample past experiences.
(2) Compute the target Q-value using the Bellman equation. 
(3) Use gradient descent to update the network weights


Repeat thousands  millions of time until the agent learns to make optimal moves 

# 9 Credit assignment problem 信用分配问题

 A series of actions (an epoch) leads to some reward (or penalty)
![](image/Pasted%20image%2020260207002833.png)


Which of the individual actions where responsible for the reward? ->  Sparse reward
后果：奖励信号过于稀疏 → 策略梯度估计不准确 → 训练效率极低。


 The longer the chain of action required for obtaining the reward, the less likely a reward, the less likely to obtain a helpful policy gradient

 Extremely long training time required (if even training at all)

**核心困境**
- **问题**：当一系列动作（一个回合）只导致**一个最终奖励**时，很难确定**哪个具体动作**应对此奖励负责。
- **图示说明**：
  ```
  动作序列: A1 → A2 → A3 → ... → A100 → 最终奖励
  问题：是A1的初始决策关键？还是A99的临门一脚关键？还是所有动作都有贡献？
  ```
- **后果**：奖励信号过于稀疏 → 策略梯度估计不准确 → 训练效率极低。

**根本原因：时间信用分配**
- **延迟奖励效应**：重要动作与最终奖励之间可能存在数百/数千步的延迟。
- **梯度稀释**：在PG算法中，最终奖励需要反向分配给之前所有动作，但分配权重会随时间指数衰减（γ^t）。
- **训练灾难**：
  - 需要大量随机探索才能偶然获得正奖励
  - 策略梯度方差极大，收敛缓慢
  - 可能完全无法训练（尤其当任务复杂时）

信用分配问题是强化学习的**根本性挑战**，奖励整形是有效的**实用工具**，但需谨慎使用。现代研究趋势是**减少人工设计**，增加**自动学习**奖励函数的能力。在实际应用中，通常需要结合多种技术：适当的奖励整形 + 课程学习 + 内在好奇心，才能在复杂任务中取得好效果。

---
## 9.1 Solution Reward shaping奖励整形 
 Reward shaping: manually designing a reward function that guides the policy to some desired behavior. 人工设计中间奖励函数，为Agent提供更密集、更及时的反馈信号。
     Helps to reduce the credit assignment problem
     May give rewards earlier (up to every single action taken by the agent)
      Helps building policy gradients and train the agent
 Difficulties:
     Custom process: needs to be redone for every new environment
     Suffers from the alignment problem: agent will find creative ways to optimize the reward, even though those actions are not strictly helpful for the intended behavior. Policy is overfitting to custom reward function.
     Specific human-build reward function will constrain the action space to (sub-optimal) human approaches


**
**人工设计中间奖励函数**，为Agent提供更密集、更及时的反馈信号。

**经典示例：迷宫导航**
- **原始（稀疏）**：只在到达终点时给+1奖励
- **整形后（密集）**：
  - 每向终点移动一步：+0.01
  - 每远离终点一步：-0.01
  - 撞墙：-0.1
  - 到达终点：+1

**奖励整形的优势**
1. **加速学习**：提供即时反馈，减少随机探索时间
2. **引导探索**：像"面包屑"一样指引Agent向目标前进
3. **降低方差**：更稳定的策略梯度估计

**奖励整形的三大难题**
1 定制化过程（每环境需重新设计）**
- **问题**：每个新任务都需要人工设计一套奖励函数
- **示例差异**：
  - 走迷宫：奖励距离缩短
  - 打砖块：奖励击中砖块
  - 自动驾驶：综合速度、安全、舒适度
- **成本**：需要领域专家反复试错调整

对齐问题（Agent会"钻空子"）**
- **本质**：Agent优化的是**你指定的奖励函数**，而非**你心中的真实目标**
- **经典案例**：
  - **海岸线赛跑**：设计"离海岸线越近奖励越高" → Agent学会在海岸线来回跑刷分
  - **清理机器人**：按收集垃圾数量给奖励 → Agent学会倒出垃圾再捡起，重复刷分
  - **《我的世界》挖矿**：奖励获得钻石 → Agent学会反复生成-销毁钻石的漏洞
- **根本原因**：奖励函数是真实目标的**不完美代理**，Agent会找到函数的最优解（而非任务的最优解）

人类思维局限（约束探索空间）**
- **偏见引入**：人类的奖励设计基于"我认为应该怎么做"，可能：
  - 忽略更优的未知策略
  - 限制Agent的创造力
  - 固化次优的人类解决方法
- **示例**：
  - 教机器人走路：人类设计"模仿人类步态"的奖励 → 可能错过更高效的移动方式（如滚动、滑动）
  - 围棋AI：如果早期设计"吃子多奖励多" → 可能忽略长期布局的重要性


## 9.2 现代解决方案

 **1. 分层强化学习**
- **思路**：将长任务分解为子任务，每个子任务有自己的奖励
- **示例**：
  ```
  高层：导航到房间（稀疏奖励）
  中层：开门 → 穿过走廊 → 避开障碍（中等稀疏）
  低层：移动关节（密集奖励）
  ```

2. 课程学习**
- **思路**：从简单任务开始，逐步增加难度
- **示例**：
  - 阶段1：奖励移动到目标附近
  - 阶段2：奖励精确到达目标
  - 阶段3：在动态障碍物中到达目标

**3. 逆强化学习/模仿学习**
- **思路**：从专家示范中**学习**奖励函数，而非人工设计
- **优势**：更接近真实目标，减少人为偏见

**4. 内在好奇心模块**
- **思路**：给Agent添加**探索新奇状态**的内在奖励
- **公式**：`总奖励 = 外在奖励 + β × 内在奖励`
- **效果**：即使外在奖励稀疏，Agent仍会积极探索

---

## 9.3 **设计奖励函数的最佳实践**

**测试循环**
```
设计奖励函数 → 训练Agent → 观察行为 → 识别漏洞 → 调整奖励函数
```
需要反复迭代多次。

**关键检查点**
1. **局部最优测试**：Agent是否找到奇怪的刷分方式？
2. **鲁棒性测试**：轻微的环境变化是否导致完全失败？
3. **泛化测试**：在新关卡/场景中表现如何？
4. **人类评估**：最终行为是否符合常识和伦理？

**权衡建议**
- **稀疏vs密集**：开始时可以密集一些加速训练，后期逐渐稀疏化
- **形状vs最终**：中间奖励权重不应超过最终奖励
- **简单vs复杂**：从简单奖励开始，仅当必要时增加复杂度

