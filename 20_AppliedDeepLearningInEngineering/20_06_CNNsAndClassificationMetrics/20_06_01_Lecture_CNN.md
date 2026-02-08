
# 1 Convolution vs. Cross-Correlation 


![](image/Pasted%20image%2020260208140434.png)



![](image/Pasted%20image%2020260208140623.png)


好的，这是一个在深度学习和信号处理中非常核心且容易混淆的概念。简单来说：

*   **数学定义上**：它们是**不同的运算**。
*   **在深度学习（CNN）中**：我们通常所说的"卷积"，**实际上指的是"互相关"**。

下面通过对比表格和直观图示进行详细解释。

---

**核心区别对比表**

| 特性 | **互相关** | **卷积** |
| :--- | :--- | :--- |
| **数学定义** | 滤波器与输入**直接滑动相乘求和** | 滤波器**先水平&垂直翻转（旋转180°）**，再与输入滑动相乘求和 |
| **公式 (2D)** | `(f ⋆ g)[i, j] = Σ_m Σ_n f[i+m, j+n] · g[m, n]` | `(f * g)[i, j] = Σ_m Σ_n f[i-m, j-n] · g[m, n]` |
| **滑动匹配方式** | **直接对齐** | **翻转后对齐** |
| **主要应用领域** | **深度学习（CNN）**、模板匹配、特征检测 | 信号/图像处理（保证结合律、交换律等数学性质）、物理学 |
| **可交换性** | **否**（`f ⋆ g ≠ g ⋆ f`） | **是**（`f * g = g * f`） |
| **在CNN中的意义** | 我们实际使用的操作。滤波器权重是通过学习得到的，翻转与否不影响其表达能力。 | 理论上的严格操作，但CNN中并不执行翻转步骤。 |

---

**一维示例图解（最关键！）**

假设我们有：
*   **输入信号**： `X = [a, b, c, d]`
*   **滤波器/核**： `K = [w, x, y]`

**1. 互相关 的计算过程（CNN实际使用的）**
滤波器 `K` **不翻转**，直接从左到右滑动，进行**点积**。

```
步骤1:  [a,  b,  c,  d]
        [w,  x,  y]            -> 输出1: a*w + b*x + c*y

步骤2:  [a,  b,  c,  d]
           [w,  x,  y]         -> 输出2: b*w + c*x + d*y
```
**结果（互相关）**： `[aw+bx+cy, bw+cx+dy]`

**2. 卷积 的计算过程（数学严格定义）**
滤波器 `K` **需要先翻转**成 `[y, x, w]`，然后再滑动点积。
```
翻转核: K_flipped = [y, x, w]

步骤1:  [a,  b,  c,  d]
        [y,  x,  w]            -> 输出1: a*y + b*x + c*w

步骤2:  [a,  b,  c,  d]
           [y,  x,  w]         -> 输出2: b*y + c*x + d*w
```
**结果（卷积）**： `[ay+bx+cw, by+cx+dw]`

**结论**：你可以清楚地看到，`互相关` 和 `卷积` 的结果**完全不同**，除非滤波器是中心对称的（例如 `[x, y, x]`）。

---

*为什么深度学习"混淆"了这两个概念？**

在卷积神经网络中，我们**有意识地选择使用互相关**，并称之为"卷积"，原因如下：

1.  **学习的权重是等价的**：
    *   卷积操作中的"翻转"是固定的数学步骤。
    *   在CNN中，滤波器 `K` 的权重是**通过训练随机初始化并学习得到的**。
    *   一个学习到的滤波器 `K`（用于互相关），**完全等价于** 一个学习到的、已经预先翻转好的滤波器 `K_flipped`（用于数学卷积）。因为网络自己会学会适应这种关系。

2.  **简化与效率**：
    *   直接实现互相关更简单、更直观（无需额外的翻转操作）。
    *   所有主流的深度学习框架（PyTorch, TensorFlow, Keras）中的 `Conv2d` 层，内部实现的都是**互相关**。

3.  **不影响功能**：
    *   无论是否翻转，该操作的核心目的——**局部特征提取**——都能完美实现。它保留了平移等变性，这是CNN的关键特性。

---

**总结与记忆要点**

*   **本质区别**：卷积需要**翻转核**，互相关**不翻转**。
*   **CNN的现实**：我们口中和代码里的"卷积"，**99.9%的情况下指的都是"互相关"**。这是一种约定俗成的"误用"，但已被整个领域接受。
*   **如何判断语境**：
    *   如果你在读**深度学习论文或代码**，提到"convolution"，指的就是**互相关**。
    *   如果你在读**信号处理、数学或物理教材**，提到"convolution"，指的是需要**翻转核**的严格数学定义。
*   **最终建议**：理解其区别，但在使用深度学习框架时，**只需记住我们实际使用的是互相关**即可。这种"混淆"不会影响你设计、训练或使用CNN模型。


# 2 Motivation: Computer Vision

 Analysis of medical images
     Skin cancer detection in skin lesion images
     Covid19 detection in CT scans of lungs
 Autonomous driving
     Reception (pedestrian detection, ...) 
    Road surface characterization

 State supervision, quality checks, security 
     Object detection
     Object localization

---

Engineering Examples
 BiGEye Project (DB Netz AG): Reduce the use of Glyphosate on 33,000km rail network 
 identify problematic vegetation on railway tracks
 prioritized maintenance in areas that require immediate attention


 Methods
 Large data lakes
 Object detection (rails, etc.)
 Image segmentation (vegetation)

![](image/Pasted%20image%2020260208140200.png)


Engineering Examples
 Structural integrity testing
 End-of-line product quality tests
 Sound classification (fault detection)

![](image/Pasted%20image%2020260208140221.png)





# 3 Filters and Shapes

So far... single-channels (f, g) 
 Extension of the formulation to 2D cases
     Data (input array) 𝑓 of size (𝑁 ,𝑁 ) 􏰫􏰬
     Filter ("kernel") 𝑔 of length (𝑀􏰫 , 𝑀􏰬 )

![](image/Pasted%20image%2020260208142114.png)

What if 𝑓 and/or 𝑔 have multiple depth dimensions?

## 3.1 2D Convolutions

### 3.1.1 Multiple Filters

 Convolutions by multiple kernels ("filters) at the same time, e.g. different edge detectors
 Number of kernels determines the number of output channels!


![](image/Pasted%20image%2020260208142939.png)


### 3.1.2 Multi-Channel Inputs

 Convolutions for volumes: 𝑛􏰱 multi-channel inputs 𝑛􏰫 × 𝑛􏰬 × (𝑛􏰲) ; single kernel; padded conv.

![](image/Pasted%20image%2020260208143032.png)

 The last dimension (depth) of the kernel has to match the number of input channels
 Defines kernels with depth equal to the input depth. Kernel does not slide into depth dimension. 
 Each input channel has its own learned weight slice within the kernel.

 Convolutions for volumes: kernel sizes are automatically spanning the input channels 𝑛_c
 Minimal example (TensorFlow): 
    input(4×4×𝑛􏰱)with𝑛􏰲 =3channels 
     single 2D kernel (3 × 3)
 Number of trainable weights: 27 (3 × 3 × 𝑛􏰱)

![](image/Pasted%20image%2020260208143951.png)

### 3.1.3 Multi-Chn. Inputs & Kernels
 Convolutions for volumes: 𝑛_c multi-channel inputs 𝑛􏰫 × 𝑛􏰬 × (𝑛􏰲) and 𝑛􏰳 multiple kernels (padded convolutions, stride 1)

 Example: 𝑛􏰳 = 2 kernels (each 2D convolution (3 × 3)), 𝑛􏰱 = 3 input channels



![](image/Pasted%20image%2020260208144051.png)


### 3.1.4 

Single channel Inputs

 Image 4 ×4, single channel (like in gray-scale or binary images)
 Single kernel 𝑘 = (3 × 3), stride 𝑠 = (1,1), zero padding 𝑝 = (1,1)


![](image/Pasted%20image%2020260208144214.png)

please note that rounding was performed for the display of numbers. Calculations can be affected by rounding errors

---

Multiple Filters
please note that rounding was performed for the display of numbers. Calculations can be affected by rounding errors

 Example: input 4 × 4, single channel; two kernels (3 × 3), stride 𝑠 = (1,1), zero padding 𝑝 = (1,1)

![](image/Pasted%20image%2020260208144247.png)

---

Multi-Channel Inputs

 Input 4 × 4 × 𝒏𝐜 = 𝟑 channels; single kernel (3 × 3), stride 𝑠 = (1,1), zero padding 𝑝 = (1,1)

![](image/Pasted%20image%2020260208144455.png)

----
Multi-Chn. Inputs & Kernels

 Input 4 × 4 × 𝒏𝐜 = 𝟑 channels; two kernels (3 × 3), stride 𝑠 = (1,1), zero padding 𝑝 = (1,1)

![](image/Pasted%20image%2020260208144616.png)


## 3.2 Quiz


 Compute the number of trainable parameters and the output dimension for following 2D convolutional layers. Assume "same" (padded) convolutions, stride 1, and no bias term
1. Task
 Input: 10 (28 × 28) grayscale images 
 2D CNN layer: three (5 × 5) kernels
Number of trainable weights: 3 ⋅ 5 ⋅ 5 = 𝟕𝟓 Output shape: 10 × 28 × 28 × 3

2. Task
 Input: 100 (600 × 800) grayscale images 
 2D CNN layer: three (5 × 5) kernels
Number of trainable weights: 32 ⋅ 5 ⋅ 5 ⋅ 3 = 𝟐𝟒𝟎𝟎 Output shape: 15 × 600 × 800 × 32


3. Task
 Input: 15 (600 × 800) RGB images 
 2D CNN layer: 32 (5 × 5) kernels



Number of trainable weights: 3 ⋅ 5 ⋅ 5 = 𝟕𝟓 Output shape: 100 × 600 × 800 × 3


## 3.3 3D Convolutions

 3D kernel is shifted along 𝑥 − 𝑦 direction AND 𝑧 direction (corresponding to the third input. Input data is 4D

 Example: MRI or CT scan data
 Take images from different angles
 Concatenate those images into 4D input data representation
 Apply 3D filters to learn representations across different camera angles

 Example: video data
 GIF: 400 × 400 pixels, 15 fps
 Time as third dimension
 Single frame (RGB) in 𝑥 − 𝑦 − 𝑐 plane
 1s of video(400 × 400 × 15 × 3) tensor

![](image/Pasted%20image%2020260208144810.png)


## 3.4 Convolution Layer: Bias

 CNN layers will allow to use bias terms
 One bias term per kernel

![](image/Pasted%20image%2020260208144844.png)


# 4 Pooling Layers





 Pooling layers reduce the size of feature maps by summarizing over sub-regions of the inputs
    output = 𝑓 subregion

 Discrete convolutions 𝑓 would compute a weighted sum over the subregion entries
 Pooling functions employ different functions 𝑓, such as
     Maximum pooling: return maximum value of subregion 
     Average pooling: return average value of subregion


![](image/Pasted%20image%2020260208144948.png)





 Pooling layers reduce the size of feature maps by summarizing over sub-regions of the inputs
 No padding applied (typically)

 Compression rate defined by pooling size and strides
     output size: (􏰇􏰴􏰞􏰜􏰴 + 1) × (􏰶􏰴􏰞􏰜􏰷 + 1) × 𝑛􏰱 􏰵􏰴 􏰵􏰷
     where 𝑛􏰸, 𝑛􏰹, 𝑛􏰱 shape of previous layer's feature map
     where 𝑘􏰸, 𝑘􏰹 shape of pooling kernel and 𝑠􏰸, 𝑠􏰹 stride of pooling kernel

 Pooling layers
     Do not carry any trainable parameters
     Reduce dimensionality of feature maps, increase robustness of feature extractors (can you say the same with half the number of words?)
     Reduce the computational efforts in the subsequent layers
     Reduce the number of parameters (when applied before densely connected layer)

![](image/Pasted%20image%2020260208145053.png)




# 5 Dense layers(fully connected layers 全连接层)


在卷积神经网络（CNN）中，**Dense Layer（全连接层）** 是指**每个神经元都与前一层的所有神经元完全相连**的层。它通常出现在CNN架构的**末端**，用于完成从提取的**空间特征**到**最终输出**（如分类标签）的映射。

---

**一、Dense Layer在CNN中的典型位置与作用**

一个经典的CNN架构流程如下：
```
输入图像 → [卷积层 + 激活 + 池化] × N → 展平层 → 全连接层 × M → 输出层
```
**1. 特征提取阶段（前端）**
- **卷积层**：使用**局部连接**和**权重共享**提取局部特征（如边缘、纹理）。
- **池化层**：降维，增强平移不变性。
- **结果**：得到一系列**特征图**，它们保留了空间结构，但每个位置的特征是高度抽象的。

**2. 分类/回归阶段（末端） **
- **展平层**：将最后的特征图"压平"成一个**一维特征向量**。
  - 例如，一个 `5×5×128` 的特征图 → 展平为 `1600×1` 的向量。
- **全连接层**：
  - 接收这个长向量作为输入。
  - 通过学习这些高级特征之间的**全局非线性组合**，完成最终任务。
  - 例如：将"车轮"、"车窗"、"车灯"等局部特征组合起来，判断是否为"汽车"。

---

 **二、Dense Layer vs. 卷积层：核心区别**

| 特性 | **卷积层** | **全连接层** |
|------|-----------|-------------|
| **连接方式** | **局部连接**：神经元只连接输入的一个小区域（如3×3） | **全局连接**：每个神经元连接前一层的所有输出 |
| **权重共享** | **是**：同一卷积核在不同位置共享参数 | **否**：每个连接都有独立参数 |
| **参数数量** | **极少**（由卷积核大小和数量决定） | **极多**（由输入和输出的维度乘积决定） |
| **空间信息** | **保留**：输入输出的都是3D张量（宽×高×通道） | **丢弃**：输入输出都是1D向量（展平后） |
| **功能角色** | **特征提取器**：检测局部模式 | **决策器**：基于全局特征进行综合判断 |

**直观比喻**：
- **卷积层**：像一群**局部侦察兵**，每个人只负责检查图像的一小块区域，并报告该区域是否有特定模式（如边缘、拐角）。
- **全连接层**：像一位**总指挥部**，它听取所有侦察兵的报告（所有局部特征），综合考虑所有信息后，做出最终决策（这是什么物体）。

---

 **三、为什么CNN末端需要Dense Layer？**

1. **整合全局信息**：
   - 卷积层提取的是**局部特征**。要识别一个物体，需要知道这些局部特征之间的**空间关系和组合方式**。
   - 全连接层能够学习到如"车轮通常在车身底部"这样的**全局上下文信息**。

2. **强大的非线性建模能力**：
   - 多个全连接层叠加，配合激活函数，可以形成非常复杂的非线性决策边界。
   - 这是将高级特征映射到最终类别得分的关键。

3. **灵活的输出维度**：
   - 可以轻松地将任意维度的展平特征，映射到所需的输出维度（如1000个类别）。

---

**四、现代CNN架构的演变趋势**

近年来，全连接层的使用在减少，原因如下：

1. **参数爆炸问题**：
   - 举例：AlexNet中第一个全连接层有 `4096×9216 ≈ 3700万` 个参数，占整个网络参数的绝大部分。
   - 容易导致**过拟合**，尤其是数据不足时。

2. **被全局平均池化替代**：
   - **全局平均池化**：对最后一个特征图的每个通道直接取平均值，得到一个通道数的向量。
     - 例如：`7×7×1024` → GAP → `1×1×1024` → 直接输入分类器。
   - **优点**：
     - **参数为0**，大大减少过拟合。
     - 强制特征图与类别直接对应，更具解释性。
   - **代表网络**：**GoogleNet、ResNet** 等现代网络大量使用GAP，或者只用1个小型全连接层。

3. **全卷积网络**：
   - 完全摈弃全连接层，全部由卷积层构成。
   - 输出可以是空间图（如分割图），而不是单个向量。
   - **代表**：FCN（全卷积网络）、U-Net等用于语义分割的网络。

---

**五、代码示例（PyTorch/Keras风格）**

```python
# 一个典型的CNN分类模型片段
model = Sequential([
    # 特征提取部分（卷积+池化）
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 过渡：从3D特征图到1D向量
    Flatten(),  # 展平层
    
    # 决策部分（全连接层）
    Dense(128, activation='relu'),  # 第一个全连接层（隐含层）
    Dropout(0.5),                  # 防止过拟合
    Dense(10, activation='softmax') # 输出层（如10分类）
])
```

---

**总结**
- **Dense Layer（全连接层）** 是CNN末端用于**高级特征综合与决策**的组件。
- 它**丢弃空间信息**，但**整合全局上下文**，将抽象特征映射到最终输出。
- 传统CNN严重依赖它，但现代架构倾向于用 **全局平均池化** 或 **1×1卷积** 来减少其参数量和过拟合风险。
- 理解它的关键在于：**卷积层是"眼睛"和"低级感官"，全连接层是"大脑"和"高级决策中心"**。两者协同，才构成了完整的视觉识别系统。


# 6 Convolutional Neural Networks


## 6.1 Basic
 So far, we considered specialized hand-crafted kernels for extracting specific features from inputs 
     Edge detectors, ...
     Smoothing, blurring, and sharpening operations, ...
 Now: how to build algorithms of convolutional operations to detect objects, such as faces? 
     What is the kernel that will differentiate person A from person B in an image?
     Hard-coded algorithms are not feasible here!
 Idea: convolutional layers that allow for training the kernel values

![](image/Pasted%20image%2020260208141114.png)


### 6.1.1 Wight sharing 

 Kernels only look at local sequences/patches of the input, e.g. 3 × 3 kernel
 The feature extraction task is independent from the feature's location in the input: to detect a vertical edge, the position of that object in the image does not matter.
 Weight sharing: share model parameters 𝜃 across input
 What is different to densely connected networks?


好的，这是一张解释 CNN（卷积神经网络）中权重共享与全连接层区别的示意图。我来为你详细解读。
图中对比了全连接层和CNN卷积层的连接方式和参数量，核心在于说明 "权重共享" 带来的巨大优势。

![](image/Pasted%20image%2020260208141132.png)




好的，这是一张解释 **CNN（卷积神经网络）中权重共享**与**全连接层区别**的示意图。我来为你详细解读。

图中对比了**全连接层**和**CNN卷积层**的连接方式和参数量，核心在于说明 **"权重共享"** 带来的巨大优势。

---

**左侧：全连接层（Densely Connected Layer）**
- **结构**：输入层的每个神经元（`x₁` 到 `x₆`，共6个）与输出层的每个神经元（`h₁` 到 `h₆`，共6个）都有一条独立的连接线。
- **参数量**：每条连接线对应一个独立的权重参数 `θ`。
- **计算**：对于这个6输入、6输出的全连接层，总权重数量为 **6 × 6 = 36** 个。
- **特点**：参数多，且每个参数**只用于一对特定的输入-输出连接**。如果输入是图像，那么学习到的特征会**绑定在特定的像素位置上**。

**右侧：CNN层（Convolutional Layer，k=3, s=1）**
- **结构**：
    - `k = 3`：卷积核大小为 3（即宽度为3）。
    - `s = 1`：滑动步长为 1。
    - 图中展示了**一个卷积核**在输入序列（`x₁` 到 `x₆`）上滑动的过程。
- **操作**：这个3维的卷积核（有3个权重参数 `θ₁, θ₂, θ₃`）从左到右滑动。在每一个位置，它都与局部的3个输入（如 `x₁, x₂, x₃`）进行**互相关**计算，生成一个输出值（如 `h₁`）。
- **参数量**：无论这个核在输入上滑动多少次、生成多少个输出值（这里是4个：`h₁, h₂, h₃, h₄`），它**始终使用同一组3个权重参数**。
- **特点**：参数极少，且同一组参数被**重复使用（共享）** 于输入的不同位置。

---

 **二、核心概念："权重共享"详解**

 **1. 什么是权重共享？**
- 在CNN中，**一个卷积核的权重参数在整个输入（如图像）上滑动并重复使用**。
- 在上图中，卷积核的 `[θ₁, θ₂, θ₃]` 在计算 `h₁`（对应输入 `x₁, x₂, x₃`）时使用，在计算 `h₂`（对应输入 `x₂, x₃, x₄`）时**再次使用同一组参数**，以此类推。
- **共享的是什么**：共享的是"**特征检测器**"。例如，一个用于检测"垂直边缘"的卷积核，无论这个边缘出现在图像的左上角还是右下角，都由**同一个核**来检测。

**2. 为什么需要权重共享？（图的核心理念）**
- **平移不变性**：图像中一个物体的特征（如边缘、纹理）的重要性与其在图像中的**位置无关**。权重共享使得CNN学习到的特征检测能力是位置无关的。
- **极大地减少参数量**：
    - **全连接层**：输入6个单元，输出6个单元 → **36个参数**。
    - **CNN层**：一个3×1的核 → **仅3个参数**（不考虑偏置），就能处理任意长度的输入并生成相应输出。
    - 在真实图像中（例如224x224的图片），全连接层的参数将是天文数字，而CNN通过一个小核（如3x3）的共享，参数量变得极小。
- **降低过拟合风险**：参数少意味着模型复杂度低，更不容易记住训练数据中的噪声，泛化能力更强。

---

三、与全连接网络的本质区别（回答图中的问题）**

| 特性 | **全连接网络** | **卷积神经网络（CNN）** |
| :--- | :--- | :--- |
| **连接方式** | 每个输入与每个输出相连 | 每个输出只与**局部输入**相连（局部连接） |
| **参数使用** | **无共享**：每个连接有独立参数 | **权重共享**：同一卷积核的参数在整个输入上重复使用 |
| **对位置的敏感度** | 高度敏感，学到的模式绑定于特定坐标 | **平移不变**，学到的模式可在任何位置被检测 |
| **参数量** | 巨大（与输入/输出尺寸的乘积成正比） | 极小（只取决于卷积核的大小和数量） |
| **适用数据** | 结构化向量数据（如表格） | 具有**空间/局部结构**的数据（如图像、序列、音频） |

**简单比喻**：
- **全连接层**：聘请了36位**不同的、固定的**专家，每位只负责检查输入中某个特定位置与输出中某个特定位置的关系。
- **CNN层**：只聘请了**3位专家（一个小组）**，但这个小组会**一起巡视**输入的每一个局部区域，检查相同的特征模式。他们走到哪，就用同一套标准检查到哪。

---

**四、总结与图中信息关联**

这张图完美地展示了CNN的两大核心思想：
1.  **局部连接**：卷积核只"看"输入的一小部分（如3个像素），而不是全部。
2.  **权重共享**：图中连接线看起来很多，但它们**颜色相同**（暗示共享同一组参数 `θ₁, θ₂, θ₃`），因此实际参数只有3个。

**最终结论**：CNN通过 **"局部连接" + "权重共享"** ，实现了用极少的参数高效处理具有空间结构的数据（如图像），并获得了关键的**平移不变性**。这正是CNN在计算机视觉领域取得巨大成功的基石，也是它与传统全连接网络最根本的区别。

### 6.1.2 detect a vertical edge, the position of that object in the image does not matter.  为什么 

您问到了一个非常核心的问题：**为什么特征提取任务与特征在输入中的位置无关？** 这其实是卷积神经网络（CNN）设计哲学的精髓。

我们可以用一个直观的比喻和严谨的数学原理来理解。

---

**核心比喻：移动的"特征探测器"**

想象你有一把专门识别"垂直边缘"的**小尺子**（这把尺子就是**卷积核**，它的权重模式让它对垂直边缘响应最强）。

1.  **在图像左上角**：你把尺子放在那里，如果那里恰好有一个垂直边缘，尺子就会"亮灯"（产生高激活值）。
2.  **把尺子移到图像右下角**：**你并没有换尺子**！你用的是**同一把**识别垂直边缘的尺子。如果右下角也有垂直边缘，它同样会"亮灯"。

**这就是"位置无关"的本质**：**同一套特征检测规则（同一组卷积核权重），被应用于图像的每一个位置。** 特征（垂直边缘）本身是重要的，至于它出现在画面的左边还是右边，对于"它是不是一个垂直边缘"这个问题来说，**无关紧要**。

---
 **从三个层面深入理解"为什么无关"**

1**视觉任务的本质需求**
图像中物体的**语义信息通常与其绝对坐标无关**。
*   **一只猫**，无论它在画面中央还是角落，它都是一只猫。
*   **一个垂直边缘**，无论是门窗的边框还是桌腿的轮廓，它都是一个垂直边缘。
我们构建的视觉系统，理应具备这种**平移不变性**——识别物体的能力不随物体位置变化而改变。CNN的权重共享机制，正是为了**内嵌**这种先验知识而设计的。

**2. 数学与计算层面的"无关"**
*   **卷积核是一个函数**：一个卷积核定义了一个从局部像素到单个数值的**映射函数** `f(·)`。
*   **权重共享即函数复用**：对于输入图像 `I`，在位置 `(i, j)` 的输出是 `f(I(i, j) 的局部区域)`，在位置 `(m, n)` 的输出是 `f(I(m, n) 的局部区域)`。**用的是同一个函数 `f`**。
*   因此，只要 `I(i, j)` 的局部区域和 `I(m, n)` 的局部区域**视觉模式相同**，函数 `f` 就会给出**相同或相近的输出**。`f` 本身不关心 `(i, j)` 和 `(m, n)` 的坐标差，它只关心输入给它的局部像素块是什么。

**3. 与全连接网络的残酷对比**
如果使用全连接层，情况会截然不同：
*   连接图像左上角区域的神经元权重，和连接右下角区域的神经元权重是**完全独立、不同**的。
*   网络必须**分别学习**"如何在左上角识别猫"和"如何在右下角识别猫"。这不仅是巨大的数据浪费，也几乎不可能实现，因为猫可能出现在训练集从未出现过的位置。
*   这就像为图像的每一个像素位置雇佣一位**专属私人侦探**，每位侦探只认识自己辖区内的猫。CNN的解决方案则是训练一批**通用侦探**（卷积核），派他们去巡视图像的每一个区域。

---


**一个具体的数值例子**

假设我们有一个极其简单的"垂直边缘检测器"卷积核：
```
Kernel = [[ 1,  0, -1 ],
          [ 1,  0, -1 ],
          [ 1,  0, -1 ]]
```
这个核会对左侧亮、右侧暗的垂直区域产生强烈响应。

**场景A：边缘在位置 (2, 2)**
在图像某处，有一个局部区域：
```
区域A = [[255, 255,   0],
         [255, 255,   0],
         [255, 255,   0]]  # 左边亮(255)，右边暗(0)
```
将核放在该区域上进行点积运算，会得到一个**很大的正数值**。

**场景B：边缘在位置 (10, 15)**
在图像的另一个完全不同的位置，有一个**一模一样的局部模式**：
```
区域B = [[255, 255,   0],
         [255, 255,   0],
         [255, 255,   0]]
```
**使用同一个核** `Kernel` 与区域B做运算，会得到**完全相同的、很大的正数值**。

**结论**：`区域A` 和 `区域B` 的**绝对坐标**(2,2)和(10,15)不同，但它们的**局部视觉模式**相同。卷积核 `Kernel` **只对模式有响应，对坐标无响应**。因此，特征提取（检测这个特定的垂直边缘模式）与位置无关。

---

总结

"特征提取与位置无关"这句话，在CNN中是由**权重共享**这一机制**强制保证**的，而不是一个偶然结果。它的合理性源于：
1.  **现实世界的先验**：视觉世界具有平移不变性。
2.  **高效的参数利用**：一个特征只需学习一次，即可到处使用。
3.  **强大的泛化能力**：即使物体出现在训练时未见过的新位置，也能被识别。

所以，**无关是因为我们通过共享权重的设计，让网络无法、也不必去关心位置**。它被迫成为一个"模式识别专家"，而非"坐标记忆专家"。这才是CNN强大泛化能力的根源之一。


## 6.2 Basic Architectures

 Convolutional neural networks are composed of stacked convolutional layers
 Stacking multiple convolutional layers allows to learn low-level and high-level features

![](image/Pasted%20image%2020260208150727.png)

 Convolutional neural networks are composed of stacked convolutional layers
 Typical structure:
     Spatial resolution decreases (convs, pooling): increasingly complex features
     Number of feature maps (semantics) increases: unfolding of multiple representations

![](image/Pasted%20image%2020260208150830.png)


Design Principles
 CNNs for classification tasks
1. Double number of filters whenever spatial resolution is halved
2. Keep kernels small (and odd-sized) for parameter efficiency
3. Use batch normalization after convolutions ( next lecture)
4. Use nonlinearities after batch normalization
5. Use global average pooling at the end to avoid fully connected layers with large parameter counts
6. Limit depth to what the data scale can support to avoid overfitting
7. Maintain consistent strides except for explicit downsampling layers


**设计原则**
- **适用于分类任务的卷积神经网络**
  1. **每当空间分辨率减半时，将滤波器数量翻倍**  
     （确保模型在更高层拥有足够的表征能力来编码更抽象的特征）
  2. **保持卷积核较小（且为奇数尺寸）以实现参数高效性**  
     （如 3×3 或 5×5 核，奇数尺寸有利于对称填充与定位）
  3. **在卷积操作后使用批归一化**（将在下一讲中介绍）  
     （加速训练并提升模型稳定性）
  4. **在批归一化后使用非线性激活函数**  
     （如 ReLU，以引入网络所需的非线性表达能力）
  5. **在模型末端使用全局平均池化，以避免参数量巨大的全连接层**  
     （将每个特征图降维为一个标量，显著减少参数并缓解过拟合）
  6. **根据数据规模限制网络深度，以避免过拟合**  
     （数据量较小时不宜使用过深网络，防止模型过度复杂）
  7. **保持步长一致，仅在下采样层中显式改变**  
     （通常默认步长为1，仅在下采样层（如池化或步长>1的卷积）中调整，以维持特征图尺寸的可控性）




