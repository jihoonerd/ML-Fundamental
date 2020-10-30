# Length and Distance

이제 내적을 이용해서 벡터공간에서의 길이와 거리, 각도등을 정의할 수 있게 된다. 이번 문서에서는 길이와 거리에 대해 다룬다.

## Length

이미 벡터의 길이를 정의하는 방법으로써 norm을 다루었다. 여기서는 내적과 norm의 관계에 집중한다. 우선, 내적을 이용하면 norm을 다음과 같이 표현할 수 있다.

$$\lVert \boldsymbol{x} \rVert \coloneqq \sqrt{ \langle \boldsymbol{x}, \boldsymbol{x} \rangle }$$

위의 간단한 식으로 내적과 norm을 연결시킬 수 있다. 하지만 물론 norm이 2-norm만 있는 것은 아니므로 모든 norm이 내적으로 유도된다고 할 수는 없다. 대표적인게 Manhattan norm(1-norm)으로 대응하는 내적표현식이 존재하지 않는다.

이후 논의를 진행하기에 앞서 거리개념에 일반적으로 적용할 수 있는 Cauchy-Schwarz 부등식을 살펴보자.

> [!NOTE]
> **Definition: Cauchy-Schwarz Inequality**
> 
> 내적공간 $(V, \langle \cdot, \cdot \rangle)$이 유도하는 norm $\lVert \cdot \rVert$는 다음의 **Cauchy-Schwartz inequality** 를 만족한다.
> $$\lvert \langle \boldsymbol{x}, \boldsymbol{y} \rangle \rvert \leqslant \lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert$$

## Distance and Metric

이제 거리에 대해 정의해보자. 거리는 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Distance**
>
> 내적공간 $(V, \langle \cdot, \cdot \rangle)$의 두 벡터 $\boldsymbol{x}, \boldsymbol{y} \in V$, $\boldsymbol{x}$와 $\boldsymbol{y}$ **거리(distance)** 는 다음과 같이 정의된다.
> $$d(\boldsymbol{x}, \boldsymbol{y}):=\|\boldsymbol{x}-\boldsymbol{y}\|=\sqrt{\langle\boldsymbol{x}-\boldsymbol{y}, \boldsymbol{x}-\boldsymbol{y}\rangle}$$

내적으로 dot product를 사용할 때에 한하여 거리를 Euclidean distance라고 한다.

> [!NOTE]
> **Definition: Metric**
>
> Metric은 보다 일반적인 정의로 다음과 같은 mapping을 **metric** 이라고 한다. Metric은 distance function으로 불리기도 한다.
> $$\begin{aligned}d: V \times V & \rightarrow \mathbb{R} \\(\boldsymbol{x}, \boldsymbol{y}) & \mapsto d(\boldsymbol{x}, \boldsymbol{y})
\end{aligned}$$

> [!WARNING]
> 길이와 비슷하게 두 벡터를정의하기 위해 norm은 충분조건이지만 내적이 반드시 정의되어야 하는 것은 아니다. 또한 내적의 정의에 따라 거리는 다르게 정의된다.

### Properties of Metric

Metric은 다음의 성질을 만족한다.

1. $d$는 positive definite하다.

  모든 $\boldsymbol{x}, \boldsymbol{y} \in V$에 대해 $ d(\boldsymbol{x}, \boldsymbol{y}) \geq 0 $이 성립하며 $d(\boldsymbol{x}, \boldsymbol{y}) = 0$이면 $ \boldsymbol{x} = \boldsymbol{y} $이다.

2. $d$는 symmetric하다.

  모든 $\boldsymbol{x}, \boldsymbol{y} \in V$에 대해 $ d(\boldsymbol{x}, \boldsymbol{y}) = d(\boldsymbol{y}, \boldsymbol{x}) $

3. Triangle inequality가 성립한다.

  모든 $\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{z} \in V$에 대해 $ d(\boldsymbol{x}, \boldsymbol{z}) \leq d(\boldsymbol{x}, \boldsymbol{y}) + d(\boldsymbol{y}, \boldsymbol{z})$가 성립한다.

## Conclusion

자주 사용하는 길이, 거리에 대한 개념을 벡터공간에서 내적과 연관지어 정의하였다. 내적과 거리는 분명 관련이 깊지만 대소관계에서 반대방향을 가진다는 것에 유의하자. 어떤 두 벡터가 매우 유사하다면 내적은 큰 값을, 거리는 작은 값을 가지게 되는 것이 두 개념의 본질적인 속성이다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.s`