# Inner Product

내적(inner product) 해석기하(analytic geometry)의 관점에서 벡터의 길이, 혹은 벡터끼리 이루는 각도나 거리를 정의하는데 유용하게 사용되며 특히 직교성(orthogonality)을 확인하는데 유용하다. 또한 symmetric과 positive definite 성질은 이후 행렬분해(matrix decomposition)를 이해하는데 있어 필요한 성질이다.

## Dot Product

엄밀한 의미에서 내적은 dot product와 구분된다. 내적은 다양한 형태의 연산을 가질 수 있으며 dot product는 내적의 한 형태이다.

Dot product는 가장 익숙한 형태의 내적으로 다음과 같이 정의된다.

$$
\boldsymbol{x}^\top \boldsymbol{y} = \sum_{i=1}^n x_i y_i
$$

일반적인 내적의 정의를 알아보기에 앞서 정의에 필요한 symmetric성질과 positive definite이라는 성질을 먼저 살펴보자.

## Symmetric and Positive Definite

$V$라는 벡터 공간에 대하여 Bilinear mapping $\Omega: V \times V \to \mathbb{R}$가 있다고 해보자. Bilinear mapping은 두 argument에 대해 각각 선형성 성립하는 mapping이다. 위의 dot product와 같이 두 벡터에 대해 각각 선형성이 성립하며 실수공간으로 mapping이 되는 것을 예로 생각해 볼 수 있다.

수식으로 표현하면 다음이 성립한다.

> [!NOTE]
> **Definition: Bilinear Mapping**
>
> 벡터공간 $V$에서의 벡터 $\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{z} \in V$와 실수 $\lambda, \psi \in \mathbb{R}$에 대해 다음이 성립하는 mapping을 **bilinear mapping** 이라고 한다.
> $$\begin{aligned}
\Omega(\lambda \boldsymbol{x}+\psi \boldsymbol{y}, \boldsymbol{z}) &=\lambda \Omega(\boldsymbol{x}, \boldsymbol{z})+\psi \Omega(\boldsymbol{y}, \boldsymbol{z}) \\
\Omega(\boldsymbol{x}, \lambda \boldsymbol{y}+\psi \boldsymbol{z}) &=\lambda \Omega(\boldsymbol{x}, \boldsymbol{y})+\psi \Omega(\boldsymbol{x}, \boldsymbol{z})
\end{aligned}$$


### Symmetric

> [!NOTE]
> **Definition: Symmetric**
>
> Bilinear mapping이 다음을 만족하면 symmetric하다고 한다.
> $$\Omega(\boldsymbol{x}, \boldsymbol{y}) = \Omega(\boldsymbol{y}, \boldsymbol{x}) \ \text{for all} \ \boldsymbol{x}, \boldsymbol{y} \in V$$

즉, bilinear mapping에 대해 순서가 상관이 없을 경우 symmetric하다고 한다.

### Positive Definite

> [!NOTE]
> **Definition: Positive Definite**
> 
> 다음을 만족하면 $\Omega$는 **positive definite**이라고 한다.
> $$\forall \boldsymbol{x} \in V \setminus \{ \boldsymbol{0} \}: \Omega(\boldsymbol{x}, \boldsymbol{x}) > 0, \ \Omega(\boldsymbol{0}, \boldsymbol{0}) = 0$$

즉, $\boldsymbol{0}$이 아닌 벡터공간 $V$의 원소 $\boldsymbol{x}$가 자기 자신에 대한 mapping $\Omega$에 대해 양의 값을 가지고 $\boldsymbol{0}$간의 $\Omega$ 결과가 실수 0일때 $\Omega$를 postive definite하다고 한다.

## Inner Product

앞서 언급된 성질인 symmetric과 positive definite을 통해 내적을 정의할 수 있다. 

> [!NOTE]
> **Definition: Inner Product**
>
> Positive definite하고 symmetric한 bilinear mapping $\Omega: V \times V \to \mathbb{R}$를 $V$에 대한 **내적(inner product)**이라고 하며 $\langle \boldsymbol{x}, \boldsymbol{y} \rangle$로 표현한다. $(V, \langle \cdot, \cdot \rangle )$는 **내적공간(inner product space)**로 정의한다.

### Example

내적과 dot product를 구분하기 위해 책에 언급된 예제를 살펴보자.

> [!WARNING]
> **Example: Inner Product That Is Not the Dot Product**
> 
> Consider $V = \mathbb{R}^2$. If we define
> 
> $$\langle \boldsymbol{x}, \boldsymbol{y} \rangle := x_1y_1 - (x_1y_2 + x_2y_1) + 2x_2y_2$$
> then $\langle \cdot, \cdot \rangle$ is an inner product but different from the dot product.
>

벡터 $\boldsymbol{x} = [x_1\ x_2]^\top$와 $\boldsymbol{y} = [y_1\ y_2]^\top$의 순서를 바꾸어도 성립함을 알 수 있으며(symmetric), 영벡터에 대해서는 0으로 mapping이 되므로(positive definite) 문제에서 정의된 mapping은 내적이다. 하지만 dot product $(x_1y_1 + x_2y_2)$는 아님을 알 수 있다.

## Symmetric, Positive Definite Matrices

내적의 정의를 통해 symmetric positive (semi)definite matrix도 유도할 수 있다. $n$차원 벡터공간 $V$에 대하여 basis가 $B = (\boldsymbol{b}_1, \ldots, \boldsymbol{b}_n)$라면 벡터공간 내 임의의 $\boldsymbol{x}, \boldsymbol{y}$는 basis의 선형결합으로 표현될 수 있다.

$$
\boldsymbol{x} = \sum_{i=1}^n \psi_i\boldsymbol{b}_i, \ \psi_i \in \mathbb{R}
$$

$$
\boldsymbol{y} = \sum_{j=1}^n \lambda_j\boldsymbol{b}_j, \ \lambda_j \in \mathbb{R}
$$

이 두 벡터로 내적을 하면 선형성 의해 다음과 같이 표현될 수 있다.

$$
\begin{aligned}
\langle \boldsymbol{x}, \boldsymbol{y} \rangle &= \left \langle \sum_{i=1}^n \psi_i \boldsymbol{b}_i, \sum_{j=1}^n \lambda_j \boldsymbol{b}_j \right \rangle \\
&= \sum_{i=1}^n \sum_{j=1}^n \psi_i \langle \boldsymbol{b}_i, \boldsymbol{b}_j \rangle \lambda_j \\
&= \hat{\boldsymbol{x}}^\top \boldsymbol{A} \hat{\boldsymbol{y}}
\end{aligned}
$$

이 때, $A_{ij}:= \langle \boldsymbol{b}_i, \boldsymbol{b}_j \rangle$이며 $\hat{\boldsymbol{x}}, \hat{\boldsymbol{y}}$는 basis $B$에 대한 $\boldsymbol{x}, \boldsymbol{y}$의 좌표이다. 이 말은 내적 $\langle \cdot, \cdot \rangle$이 행렬 $\boldsymbol{A}$에 의해 유일하게 결정된다는 것을 의미한다. 따라서 해당 내적의 symmetric 여부는 $\boldsymbol{A}$가 결정하며 내적의 positive definiteness는 $\forall \boldsymbol{x} \in V \setminus \{\boldsymbol{0}\}: \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x} > 0$를 만족하여야 한다. 위 식에서 등호가 성립하는 경우, $\boldsymbol{A}$를 **positive semidefinite**이라고 한다.

## Inner product for a Real-valued and Finite-dimensional Vector Space

유한차원의 실수 벡터공간 $V$의 ordered basis $B$에 대해 다음이 성립한다.

* 내적 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$, 즉 벡터공간 $V$의 원소 $\boldsymbol{x}, \boldsymbol{y}$에 대해 $\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \hat{\boldsymbol{x}}^\top \boldsymbol{A} \hat{\boldsymbol{y}}$이 정의되는 것은 행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$가 symmetric하고 positive definite한 것과 동치이다.

또한 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$가 symmetric하고 positive definite하다면 다음이 성립한다.

* $\boldsymbol{A}$의 kernel은 $\boldsymbol{0}$뿐이다.
  
  Positive definite에 의해 영벡터가 아닌 모든 벡터에 대해 $\boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x} > 0$이다. 따라서 kernel이 될 수 있는건 영벡터 뿐이다.

* $\boldsymbol{A}$의 대각성분 $a_{ii}$는 모두 양수이다.
  
  대각성분은 $i$번째 basis로 $\boldsymbol{e}_i$의 자기 자신에 대한 내적이다. 따라서 $a_{ii}$는 $\boldsymbol{e}_i^\top \boldsymbol{A} \boldsymbol{e}_i$를 계산해 얻을 수 있으며 해당 값은 positive definite에 의해 양수가 된다.


## Conclusion

내적, symmetric, positive definite에 대해 알아보았다. 이 개념들은 선형대수학에서 해석기하와 행렬분해를 다룰 때 계속 등장하는 개념으로 잘 알아둘 필요가 있다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.