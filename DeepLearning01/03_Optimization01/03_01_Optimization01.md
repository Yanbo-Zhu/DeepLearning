
# 1 Gradient Descent


![[Pasted image 20251101211925.png]]

![[Pasted image 20251101212325.png]]


| 项目         | 结果                                        |
| ---------- | ----------------------------------------- |
| 最小值        | (\theta^\star)                            |
| 梯度         | (2\alpha_i(\theta_i-\theta_i^\star))      |
| Hessian    | (2\mathrm{diag}(\alpha_1,\dots,\alpha_d)) |
| 凸性         | 严格凸                                       |
| 收敛条件（梯度下降） | (0<\eta<\frac{1}{2\max_i\alpha_i})        |
![[Pasted image 20251101212754.png]]



## 1.1 Minimizer 

![[Pasted image 20251101212209.png]]


## 1.2 Gradient (componentwise)


![[Pasted image 20251101212227.png]]

设梯度为零可得最优解：




## 1.3 Hessian and convexity


![[Pasted image 20251101212355.png]]


## 1.4 Interpretation as a weighted squared norm


![[Pasted image 20251101212420.png]]

也就是说，它是一个加权的平方距离（每个坐标有不同的重要程度）。


## 1.5 Gradient descent update and stability


![[Pasted image 20251101212447.png]]

![[Pasted image 20251101212615.png]]


# 2 Gradient descent (Special )


![[Pasted image 20251101211925.png]]

![[Pasted image 20251101214017.png]]


C ist the previous system  the parameter before the update 

![[Pasted image 20251101214422.png]]


![[Pasted image 20251101220101.png]]


# 3 Gradient Descent (General Case)


![[Pasted image 20251101221249.png]]