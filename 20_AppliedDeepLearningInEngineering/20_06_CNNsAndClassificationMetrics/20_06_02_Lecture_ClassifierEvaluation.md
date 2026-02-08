
# 1 Positives and Negatives

 The positive class is the existence what you are trying to detect or predict.

 Examples:
 Production: end-of-line testing:  find faulty products. 
Positive class: faulty product 

 Health monitoring: fault detection for machines.
Positive class: machine malfunction 

 Spam filter: removing spam emails from the inbox. 
Positive class: spam email


 In most cases, the correct prediction of the positive class has highest priority  In most cases, missing a positive class sample is the most harmful
 In most cases, falsely predicting the positive class is unwanted but acceptable



# 2 Measuring Classification Performance


 Setting: binary classification problem
Straight-forwardmetric: accuracy acc= 􏰿􏱀􏰪􏰿􏱁 􏰿􏱀􏰪􏰿􏱁􏰪􏱂􏱀􏰪􏱂􏱁
ratio of correctly predicted samples


 TP: number of true positives
 FP: number of false positives
 TN: number of true negatives
 FN: number of false negatives
𝑦 = 1, 𝑦􏱃 = 1 𝑦 = 0, 𝑦􏱃 = 1 𝑦 = 0, 𝑦􏱃 = 0 𝑦 = 1, 𝑦􏱃 = 0

 Positives/negatives: depends on the application and the perspective
Typically: positive class is the quantity you want to predict / detect (deviating from the normal)

 Example: Structural health monitoring, prediction of ball bearing failure. 
 Positive class: failure exists
 Negative class: no failure

![](image/Pasted%20image%2020260208152317.png)




# 3 Classification Metrics

 Choice of the metric depends on the very specific use case
 Test edge cases (the best / worst result you can think of for the given task) and evaluate your chosen metric. Does it behave as desired?


![](image/Pasted%20image%2020260208152541.png)

## 3.1 Confusion Matrix


 Classifier output evaluation (here: binary classification)
 Display of prediction quality for a labeled data set (typically validation data set)
 Different variant of the same matrix: display of percentages. (sum amounts to 1)
 Most classification metrics are derived from the confusion matrix or can be visualized using it.

![](image/Pasted%20image%2020260208152614.png)

## 3.2 Recall and Precision

Fraction of predicted positives that are
actually correct
Large precision: very small false positives errors committed by the model
"Don't be wrong when predicting a positive"

Fraction of positives correctly predicted
▪ Large recall: very few positive samples misclassified as the negative class
"Make sure to predict all positives correctly"

![](image/Pasted%20image%2020260208152657.png)


 Recall and precision are typically competing metrics:
 One can build models that either ...

 favor precision (not looking at the negative class, allowing FN)
 Very conservative model (do not make a FP error!)

 favor recall (not looking at the false positives at all)
 Very aggressive model, allows for dummy mode


![](image/Pasted%20image%2020260208152752.png)

## 3.3 F1 Score: Combining Precision and Recall

▪ Aim: build a model that optimizes both recall and precision
▪ F1 score: harmonic mean between recall and precision 𝐹1 =
![](image/Pasted%20image%2020260208152858.png)

# 4 From binary to probabilistic evaluation


Re-Thinking (Binary) Classification
 So far, we discussed predictions to be correct or incorrect
 But: our neural classifiers use sigmoid / softmax activations in the output layers to return probabilities 𝑦􏱃 = 𝑃 𝑦 = 1 𝑥)

How to get from 𝑃 𝑦 = 1 𝑥) to true/false positives/negatives?

 Define a decision probability decision threshold 𝑦􏰕􏰖􏰗􏰉􏰘􏰖 such that 𝑦􏱃=􏱊0if𝑃 𝑥 <𝑦􏰕􏰖􏰗􏰉􏰘􏰖
1if𝑃 𝑥 ≥𝑦􏰕􏰖􏰗􏰉􏰘􏰖

 What is the best threshold, may the classifier perform better for 𝑦􏰕􏰖􏰗􏰉􏰘􏰖 ≠ 0.5?

![](image/Pasted%20image%2020260208154118.png)


## 4.1 Receiver Operating Characteristic Curve (ROC)

 Graphical display for the tradeoff between true positive rate (TPR) and false positive rate (FPR)

 Evaluate TPR and FPR for any possible 𝑦􏰕􏰖􏰗􏰉􏰘􏰖 and arrive at ROC curves


![](image/Pasted%20image%2020260208154150.png)


这张图解释了**ROC曲线（接收者操作特征曲线）**的核心概念。它本质上是一个**模型性能可视化工具**，用于展示分类模型在不同判断阈值下，**识别能力**与**误报率**之间的权衡。

---

**一、核心概念解读**

**1. 坐标轴含义**
*   **X轴：假正率** - **FPR = FP / (FP + TN)**
    *   含义：**在所有实际为负的样本中，被模型错误地判为正的比例。**
    *   我们希望它**越低越好**（越靠近0越好）。
*   **Y轴：真正率** - **TPR = TP / (TP + FN)**
    *   含义：**在所有实际为正的样本中，被模型正确找出的比例。** 也称为**召回率**。
    *   我们希望它**越高越好**（越靠近1越好）。

**2. 图中关键点与线的含义**
*   **点 `(0, 1)`**：**完美模型**
    *   FPR = 0（没有误报），TPR = 1（全部找出）。现实中几乎不存在。
*   **曲线 `model 1` 与 `model 2`**：
    *   每个点对应一个**分类阈值**（`y_thresh`）。从左下到右上移动，意味着**阈值从严格（如0.9）逐渐放宽（如0.1）**。
    *   **曲线越向左上角凸起，模型性能越好**。因此图中 `model 1` 的性能优于 `model 2`。
    *   ROC曲线展示的是模型**在不同误报容忍度下，能达到的最佳识别率**。
*   **对角线 `random guessing`**：
    *   代表一个**没有任何区分能力**的模型（如随机抛硬币），其TPR永远等于FPR。
    *   这是性能的**基准线**。任何有意义的模型曲线都应位于这条线上方。

**3. 极端点：`(0, 0)` 与 `(1, 1)`**
*   **点 `(0, 0)`**：**全负模型**
    *   对应 **`y_thresh = 1.0`**。模型对所有样本都预测为"负类"。
    *   因此，TPR = 0（没有找出任何正例），FPR = 0（也没有误报任何负例）。
*   **点 `(1, 1)`**：**全正模型**
    *   对应 **`y_thresh = 0.0`**。模型对所有样本都预测为"正类"。
    *   因此，TPR = 1（找出了所有正例），但FPR = 1（所有负例也都被误报为正例）。

---

 **二、如何生成ROC曲线？**
流程如下：
1.  模型对每个样本输出一个**概率得分**（例如，属于正类的概率在0到1之间）。
2.  设定一个**阈值** `y_thresh`，将得分高于此阈值的样本预测为正类，低于的为负类。
3.  根据这个阈值下的预测结果，计算一组 **(FPR, TPR)** 坐标。
4.  **遍历所有可能的阈值**（例如从1.0逐步降到0.0），计算得到一系列坐标点。
5.  将这些点连接起来，就形成了ROC曲线。



## 4.2 **核心评价指标：AUC** ： Area Under the (ROC) Curve

 Which model is better on average across all possible decision thresholds?
 Compute the integral of the ROC


虽然图中未标注，但ROC曲线最常用的总结性指标是 **AUC（曲线下面积）**。
*   **AUC = 1.0**：完美模型（曲线贴合左上边）。
*   **AUC = 0.5**：随机模型（曲线就是对角线）。
*   **AUC 在 0.5 到 1.0 之间**：模型具有区分能力，值越大越好。
*   **AUC < 0.5**：模型比随机猜测还差，可能将正负类预测反了。

**AUC的优点**：它是一个**单一标量**，综合了模型在所有可能阈值下的表现，且对类别分布不敏感（在数据平衡时）。


 Edge cases:
 Perfect model (TPR=1, FPR=0): 
 Random guessing (TPR=FPR=0.5):

 A model with higher potential has a higher AUC than another model

 Use case:
 Comparing the performance among different classifiers
 Comparing the performance of a classifier for different classes

![](image/Pasted%20image%2020260208155749.png)





## 4.3 Precision-Recall Curve

 Precision-recall curve (PRC): tradeoff between precision and recall for different decision thresholds
== ROC is only suited for balanced data, PRC is also working for imbalanced data==

 The longer a large precision can be sustained, the better the model. Note that the curve does not necessarily decrease monotonically with recall

 AUC-PRC can be computed analogously

 Optimal choice of probability threshold  𝑦threshfor specific use case (either favoring high precision or recall), e.g. max𝑦thresh 𝐹1


![](image/Pasted%20image%2020260208155851.png)


**一、核心概念：什么是PR曲线？**

**PR曲线**是用来可视化分类模型在**不同判定阈值下**，其**精确率**与**召回率**之间权衡关系的图表。

- **X轴：召回率** - `Recall = TP / (TP + FN)`
  - **含义**：在所有**实际为正的样本**中，被模型正确找出的比例。
  - **关注点**："我找得全不全？"（避免漏报）。
- **Y轴：精确率** - `Precision = TP / (TP + FP)`
  - **含义**：在所有**被模型预测为正的样本**中，真正为正的比例。
  - **关注点**："我找得准不准？"（避免误报）。

**核心关系**：当调整分类阈值（例如，将判定为正类的概率阈值从0.9降到0.1）时：
- **召回率会上升**（因为标准放宽，能找出更多真实正例）。
- **精确率通常会下降**（因为标准放宽，混入的假正例也变多）。

**三、如何解读PR曲线？**
1.  **曲线位置**：**曲线越靠右上角（高精确率、高召回率）越好**。
2.  **基线对比**：
    - 随机猜测模型的PR曲线是一条**水平线**，其高度等于数据中正例的比例（`正例数 / 总样本数`）。例如，如果正例占比1%，那么随机模型的精确率基线就是0.01。
    - 任何有意义的模型，其PR曲线都应**显著高于这条基线**。
3.  **"曲线形状"的要点**：
    - "模型能**在多高的召回率水平上，依然保持较高的精确率**"。例如，一个用于癌症筛查的模型，我们希望即使在召回率达到90%（找出90%的病人）时，精确率还能保持在80%以上（即报警案例中80%是真病人）。
    - **注意**：PR曲线**不一定是单调递减的**，它可能出现波动。这是因为当阈值变化时，TP、FP、FN的变化并非完全线性。

---

**四、量化指标：AUC-PRC**

与ROC曲线有AUC类似，PR曲线也有**曲线下面积**。
- **AUC-PRC 值在 0 到 1 之间**。
- **AUC-PRC 越接近 1，模型综合性能越好**。
- 在不平衡数据中，**AUC-PRC 比 AUC-ROC 更能区分模型的优劣**，因为它对正类的预测错误更敏感。

---

 **五、如何根据PR曲线选择最佳阈值？**

没有" universally best"的阈值，选择取决于**具体业务需求**：
- **需要高精确率的场景**（宁可漏报，不可误报）：
  - **例子**：推荐系统推送、法律审判、高端商品推荐。
  - **策略**：在PR曲线上，选择**精确率最高**的点对应的阈值。这通常对应较低的召回率。
- **需要高召回率的场景**（宁可误报，不可漏报）：
  - **例子**：癌症筛查、信用卡盗刷预警、关键设备故障预测。
  - **策略**：在PR曲线上，选择**召回率最高**的点对应的阈值。这通常对应较低的精确率。
- **需要平衡的场景**：
  - **策略**：可以选择**F1分数最高**的点（F1是精确率和召回率的调和平均数），或者选择PR曲线上最靠近右上角 `(1, 1)` 的点。

**实际操作**：
1.  使用模型在验证集上预测的概率。
2.  遍历所有可能的阈值，计算对应的`(召回率, 精确率)`点，绘制PR曲线。
3.  根据业务目标，在曲线上选取最符合需求的点，并记录其对应的阈值 `y_thresh`。
4.  将此阈值应用于生产环境。



总结**
PR曲线是评估分类模型（尤其是面对**不平衡数据**和**关注正类性能**时）的**强大工具**。它通过可视化**精确率**与**召回率**的权衡，帮助我们：
1.  **客观评估**模型在识别关键少数类上的真实能力。
2.  **量化比较**不同模型（通过AUC-PRC）。
3.  **科学决策**，根据业务风险偏好选择最佳分类阈值。

当你的数据中正例比例小于20%时，请优先使用PR曲线进行分析。


# 5 **ROC 不适用于高度不平衡数据, 用 PRC**

*   **问题**：当负样本（多数类）数量远超正样本（少数类）时，FPR的分母 `(FP+TN)` 会非常大。即使有少量误报（FP），计算出的FPR也会很小，这会使ROC曲线看起来**过于乐观**，无法真实反映模型在识别少数类上的实际困难。
*   **例子**：在10000个样本中（9900个负例，100个正例），即使模型将200个负例误报为正（FP=200），FPR也仅为 `200/9900 ≈ 0.02`，看起来很低，但事实上模型可能漏掉了大量正例（TPR很低）。
*   **替代方案**：对于不平衡数据，应优先查看 **PR曲线（精确率-召回率曲线）**，因为它聚焦于正类的预测质量，对类别分布更敏感。

**总结**
ROC曲线是评估二分类模型性能的**经典工具**，它通过**可视化**TPR与FPR的权衡关系，并利用**AUC**进行量化评估。其核心思想是：**一个好的模型应该能够在尽可能少误报（低FPR）的情况下，尽可能多地找出正例（高TPR）**。但在处理类别高度不平衡的数据时，需要谨慎解读，并考虑使用PR曲线作为补充。


## 5.1 为什么PR曲线比ROC曲线更适合不平衡数据？
这是最关键的一点，也是PR曲线最重要的价值。

ROC曲线的局限性（在高度不平衡数据中）：
- ROC的X轴是假正率，其分母是(FP + TN)，包含了大量的真负例。
- 当负例（多数类）数量极大时，即使模型误报了不少负例，计算出的FPR也会因为分母巨大而显得很小，导致ROC曲线过于乐观，无法敏感反映模型在识别少数类（正例）上的真实能力。

PR曲线的优势：
- PR曲线的两个指标（精确率、召回率）完全聚焦于正类的预测表现。
- 召回率的分母是全部真实正例。
- 精确率的分母是全部预测正例。
- 两者都不直接考虑真负例。因此，当数据极度不平衡（正例很少）时，PR曲线能更敏感、更真实地揭示模型在识别关键少数类上的性能优劣。模型哪怕多犯一点错误，精确率就会明显下降。

结论：对于欺诈检测、疾病筛查、缺陷检测等正例极少但至关重要的场景，PR曲线是比ROC曲线更可靠、更严格的评估工具。

# 6 Imbalanced Classes 


▪ Data imbalance (or class imbalance): one class label is severely over-represented (majority class) in the data set, while the minority class is the positive class
▪ Examples:
    ▪ Credit card frauds: 1 out of 100
    ▪ Machine faults: component failure after 10x operational hours
    ▪ Brake system vibrations (see homework no 1): 3-5% noise occurrence
▪ Fitting any model without consideration of the class imbalance will lead to
    ▪ Dummy models: predicting the majority class with 𝑃 𝑥 = 1.0
    ▪ Large false negatives rate

![](image/Pasted%20image%2020260208161814.png)

## 6.1 Strategies for Data Imbalance

▪ Different classification metrics
    ▪ FNR, TPR, F1
    ▪ Weighted scores (weight = inverse of occurrence rate)
    ▪ Matthews' correlation coefficient MCC. MCC=0 (random guessing), MCC=1.0 (best)
    ▪ PRC, AUC-PRC
▪ Data treatment: Modify class distributions (attention!!!) – not recommended
    ▪ Oversampling: insert copies (or variations) of the positive class into the data set
    ▪ Undersampling: remove samples from the majority class


![](image/Pasted%20image%2020260208161903.png)



# 7 Imbalanced Classes 例子 

场景还原**

- **数据**：共10000个样本
  - **负例**（多数类）：9900个（例如，非欺诈交易）
  - **正例**（少数类）：100个（例如，欺诈交易）

- **假设模型的预测结果如下**：
  - **真正例**：`TP = 20` （成功找出了100个欺诈交易中的20个）
  - **假正例**：`FP = 200` （将200个正常交易误判为欺诈）
  - **真负例**：`TN = 9900 - 200 = 9700` （正确识别了9700个正常交易）
  - **假负例**：`FN = 100 - 20 = 80` （漏掉了80个欺诈交易）

---

**计算指标**

1.  **假正率**：`FPR = FP / (FP + TN) = 200 / (200 + 9700) = 200 / 9900 ≈ 0.02`
    - **解读**：在所有的正常交易中，只有2%被误报了。**从FPR看，模型"误伤率"很低，似乎非常"谨慎"。**

2.  **真正率**：`TPR = TP / (TP + FN) = 20 / (20 + 80) = 20 / 100 = 0.20`
    - **解读**：在所有的欺诈交易中，模型只成功找出了20%。**从TPR看，模型漏掉了80%的欺诈，识别能力极差。**

---

**为什么会产生这种"矛盾"的感知？**

因为 **FPR 和 TPR 的关注分母不同**：
- **FPR的分母是 `9900`（全部负例）**，这是一个巨大的数字。即使犯了 `200` 个错误，**相对于庞大的基数来说，比例依然显得很小（2%）**。这给了我们一种"模型很精准"的**错觉**。
- **TPR的分母是 `100`（全部正例）**，这是一个很小的数字。即使只犯了 `80` 个错误（漏报），**相对于微小的基数来说，比例已经高达80%**。这直接暴露了模型的**严重缺陷**。

 
 **用现实比喻来理解**

想象一家医院用AI筛查一种罕见病（发病率1%）：
- **负例（健康人）**：9900人
- **正例（病人）**：100人
- **模型表现**：
  - 把200个健康人误诊为病人（让他们虚惊一场，做进一步检查）。
  - 只找出了20个真正的病人，漏掉了80个病人（让他们延误治疗）。

**医院主任如果只看"误诊率"**：`200 / 9900 ≈ 2%`，他会觉得"AI误诊率才2%，不错！"
**但事实上**：**80%的真正病人都被漏掉了！这是一个医疗灾难。**

---

 **ROC曲线在此场景下的局限性**

在这个例子中，即使模型性能很差（TPR=0.2），但因为FPR很低（0.02），这个点 `(0.02, 0.20)` 在ROC图上仍然会**靠近左侧（看起来不错）**，而不会像TPR低所暗示的那样靠近底部。

**ROC曲线的X轴（FPR）对多数类（负例）的数量过于敏感**，导致在不平衡数据中，模型只要在多数类上稍好一点，就能获得一个很低的FPR，从而"美化"了曲线。

---

 **结论与解决方案**
1.  **结论**：在不平衡数据中，一个很低的FPR**不代表模型好**，它可能只是因为负例基数太大，稀释了错误比例。与此同时，模型的TPR（识别正例的能力）可能非常低，这才是关键问题。
2.  **解决方案**：在这种情况下，应主要关注：
    *   **精确率**：`P = TP / (TP + FP) = 20 / (20+200) ≈ 9%`。这告诉我们，在所有被模型报警的案例中，只有9%是真的欺诈。这揭示了报警的**可信度极低**。
    *   **召回率**：即TPR，20%，它直接告诉我们漏掉了多少关键案例。
    *   **F1分数**：精确率和召回率的调和平均数，能更好地综合评估。
    *   **PR曲线**：绘制**精确率 vs 召回率**的曲线。在不平衡数据中，PR曲线对模型性能的变化更为敏感，是比ROC曲线**更可靠的评估工具**。

所以，回到您的原问题：**为什么可能漏掉大量正例？**
**答案**：因为模型可能过于"保守"（或能力不足），为了避免在庞大的负例上犯错（导致FP），它选择"少报警"，而这直接导致它错过了大量真正的正例（导致FN）。ROC曲线中的FPR指标由于分母巨大，掩盖了这种保守策略在正例识别上的灾难性失败。




