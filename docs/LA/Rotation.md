# Rotation

이 문서에서는 회전변환에 대해 다룬다. 수학적 의미와는 별개로 단어가 주는 느낌 때문에 마치 회전변환은 "선형"변환이 아니라는 느낌을 받을 수 있지만 회전변환도 선형변환의 일종이다. 앞서, 선형변환의 특징 중 하나로 길이와 각도의 보존이 언급되었었는데 어떤 공간을 회전한다고해서 길이나 이루고 있는 각도가 변하지는 않을 것이므로 선형변환의 특성을 만족한다는 것도 직관적으로 생각해볼 수 있다.

## Automorphism

여기서 이야기하는 선형변환은 수학적으로 Euclidean 벡터 공간에서의 automorphism 속성을 갖는 선형변환이다. 

> [!NOTE]
> **Definition: Automorphism**
> 
> 어떤 mapping $f: X \rightarrow Y$가 다음의 성질을 만족할 때 해당 변환을 **automorphism**이라고 한다.
> * 정의역과 공역이 같다. 즉, $X = Y$이다.
> * 모든 구조를 보존하는 변환(동형 사상)이 존재한다.

우리가 다루는 Euclidean 벡터 공간에서 어떠한 회전변환을 한다고 하더라도 공간이 달라지지는 않는다. 즉, 변환 전과 변환 후의 공간은 같다. 다만 표현하는 방식에 있어 회전한 임의의 $\theta$만큼의 차이가 발생할 뿐이다. 그리고 정의에서 언급된 구조란 대수적인 구조(algebraic structure)를 의미하며 변환 전후에 대수적 구조가 보존되어 앞서 언급된 벡터의 길이라든지 두 벡터가 이루는 각도같은 개념이 변하지 않는 것이다.

## Application: Robotics

Robotics분야에서는 이러한 회전변환이 빈번하게 사용된다.(물론 실제 계산과정은 framework레벨에서 처리되는 경우가 대부분이겠지만) 예를 들어, 로봇이 어떠한 물체를 집어서 특정 위치에 가져다 놓는 작업을 한다고 해보자. 대상물이 늘 정방향으로 놓여있는다는 보장이 없으므로 물체를 잡고 놓는 과정에서 시각적인 정보를 통해 물체의 방향을 파악하는 과정이 필요하다. 또한 관절의 회전과 같은 작업도 내부적으로는 회전에 관한 변환이 빈번하게 발생하는 과정이다.

## Roataions in $\mathbb{R}^{2}$

2차원 공간이 가장 익숙한 형태인 $\left\{\boldsymbol{e}_{1}=\left[\begin{array}{l}1 \\ 0\end{array}\right], \boldsymbol{e}_{2}=\left[\begin{array}{l}0 \\ 1\end{array}\right]\right\}$의 기저로 표현되어 있다고 하자.

$\theta$만큼 회전변환한 좌표계를 표현하기 위해 아래 그림을 참조하자.

<figure align=center>
<img src="assets/images/LA/Fig_3.16.png" height=50% width=50% />
<figcaption>Fig 3.16: Rotations of the standard basis in $\mathbb{R}^2$ by an angle $\theta$</figcaption>
</figure>

기저벡터를 $\theta$만큼 회전시켰을 떄의 좌표변화는 삼각함수로 표현이 가능하다는 것을 볼 수 있고 길이나 두 기저벡터가 이루는 각도의 변화가 없음을 쉽게 확인할 수 있다.

각 기저에 대한 변환은 Fig 3.16에 의해 다음과 같이 얻을 수 있다.

$$
\Phi\left(\boldsymbol{e}_{1}\right)=\left[\begin{array}{c}
\cos \theta \\
\sin \theta
\end{array}\right], \quad \Phi\left(\boldsymbol{e}_{2}\right)=\left[\begin{array}{c}
-\sin \theta \\
\cos \theta
\end{array}\right]
$$

따라서 rotation matrix는 다음과 같다.

$$
\boldsymbol{R}(\theta)=\left[\Phi\left(\boldsymbol{e}_{1}\right) \quad \Phi\left(\boldsymbol{e}_{2}\right)\right]=\left[\begin{array}{cc}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{array}\right]
$$

## Rotations in $\mathbb{R}^{3}$

3차원에서의 회전을 생각해보자. 3차원에서의 회전은 2차원에서의 회전을 각각의 축에 적용한 결과로 나누어 생각할 수 있다.

예를 들어 $\boldsymbol{e}_1$ 축에 대해서 회전을 하게되면 $\boldsymbol{e}_1$의 성분은 보존하고 나머지에 대해 회전이 적용된다. 이러한 회전변환 $\boldsymbol{R}_{1}(\theta)$는 다음과 같다.

$$
\begin{aligned}
\boldsymbol{R}_{1}(\theta)&=\left[\Phi\left(\boldsymbol{e}_{1}\right) \quad \Phi\left(\boldsymbol{e}_{2}\right) \quad \Phi\left(\boldsymbol{e}_{3}\right)\right]\\
&=\left[\begin{array}{ccc}
1 & 0 & 0 \\
0 & \cos \theta & -\sin \theta \\
0 & \sin \theta & \cos \theta
\end{array}\right]
\end{aligned}
$$

같은 원리로 $\boldsymbol{e}_2$와 $\boldsymbol{e}_3$에 대해 적용하면 각각의 축에 대해 다음과 같은 회전변환 $\boldsymbol{R}_2(\theta)$와 $\boldsymbol{R}_3(\theta)$를 얻게 된다.

$$
\boldsymbol{R}_{2}(\theta)=\left[\begin{array}{ccc}
\cos \theta & 0 & \sin \theta \\
0 & 1 & 0 \\
-\sin \theta & 0 & \cos \theta
\end{array}\right]
$$

$$
\boldsymbol{R}_{3}(\theta)=\left[\begin{array}{ccc}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{array}\right]
$$

## Rotations in $n$ Dimensions

$n$차원 공간에서의 회전으로 일반화를 해보자. 3차원 회전을 생각해보면 하나의 축을 고정하고 평면에 대한 회전으로 접근했었다. $n$차원에 대한 적용도 다르지 않다. $n-2$개의 차원을 고정하고 평면에서의 회전으로 바라보면 된다. 일반화된 식이다보니 표현이 조금 더 복잡해졌다는 차이가 있을 뿐이다.

> [!NOTE]
> **Definition: Givens Rotation**
> 
> Euclidean 벡터 공간 $V$가 $n$차원일 때, automorhism 성질을 갖는 변환 $\Phi: V \rightarrow V$의 변환행렬은 다음과 같다.
> $$
\boldsymbol{R}_{i j}(\theta):=\left[\begin{array}{ccccc}
\boldsymbol{I}_{i-1} & \boldsymbol{0} & \cdots & \cdots & \boldsymbol{0} \\
\boldsymbol{0} & \cos \theta & \boldsymbol{0} & -\sin \theta & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{0} & \boldsymbol{I}_{j-i-1} & \boldsymbol{0} & \boldsymbol{0} \\
\boldsymbol{0} & \sin \theta & \boldsymbol{0} & \cos \theta & \boldsymbol{0} \\
\boldsymbol{0} & \cdots & \cdots & \boldsymbol{0} & \boldsymbol{I}_{n-j}
\end{array}\right] \in \mathbb{R}^{n \times n} \\
$$
> for $1 \leqslant i \leq j \leqslant n$ and $\theta \in \mathbb{R}$.
> 위의 변환행렬 $\boldsymbol{R}_{i j}(\theta)$를 **Givens rotation**이라 한다.

## Properties of Rotations

회전은 다음과 같은 유용한 성질을 갖는다.

* 회전은 거리를 보존한다. 즉, $\lVert \boldsymbol{x}-\boldsymbol{y} \rVert = \lVert \boldsymbol{R}_{\theta}(\boldsymbol{x})-\boldsymbol{R}_{\theta}(\boldsymbol{y}) \rVert$이 성립한다.
* 회전은 각도를 보존한다. $\boldsymbol{x}$와 $\boldsymbol{y}$가 이루는 각도와 $\boldsymbol{R}_{\theta} \boldsymbol{x}$, $\boldsymbol{R}_{\theta} \boldsymbol{y}$가 이루는 각도는 같다.
* 3차원 이상에서의 회전변환은 commutative하지 않다. 즉, $\boldsymbol{R}(\phi) \boldsymbol{R}(\theta) \neq \boldsymbol{R}(\theta) \boldsymbol{R}(\phi)$이다. $\boldsymbol{R}(\phi) \boldsymbol{R}(\theta) = \boldsymbol{R}(\theta) \boldsymbol{R}(\phi)$는 2차원에서만 성립한다.

## Conclusion

이번 문서에서는 회전변환에 대해 다루었다. 회전변환은 선형변환 중 특정 성질을 만족하는 변환으로 실제 application에서 유용하게 사용되는 개념이다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.