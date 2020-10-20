# Linear Mapping(WIP)

선형대수학의 무대는 벡터공간이다. 벡터공간은 벡터끼리의 덧셈, 그리고 스칼라와 벡터의 곱셈이 다시 벡터공간의 원소임이 보장된다. 머신러닝에서는 벡터에 대해 다양한 변환을 하게 되는데 이러한 변환을 하더라도 변환된 공간에서 벡터공간의 기본 성질이 성립해야 유용하게 사용할 수 있다. 이번 포스팅에서는 벡터공간의 변환인 선형사상(Linear Mapping)을 다룬다.


벡터공간 $V$, $W$에 대해 변환(mapping) $\Phi: V \to W$이 다음을 만족하면 선형사상(Linear Mapping)이다.

$$
\forall \boldsymbol{x}, \boldsymbol{y} \in V, \forall \lambda, \psi \in \mathbb{R}: \Phi(\lambda \boldsymbol{x} + \psi \boldsymbol{y}) = \lambda \Phi(\boldsymbol{x}) + \psi \Phi(\boldsymbol{y})
$$

선형대수학에서는 이러한 선형사상을 행렬로 표현할 수 있다.

예를 들어 $n$개의 basis $B = (\boldsymbol{b}_1, \ldots, \boldsymbol{b}_n)$를 가지는 벡터공간 $V$와 $m$개의 basis $C = (\boldsymbol{c}_1, \ldots, \boldsymbol{c}_m)$를 가지는 벡터공간 $W$을 생각해보자. 이 때 선형사상 $\Phi: V \to W$는 다음과 같이 나타낼 수 있다.

> For $j \in \{1, \ldots, n\}$,
> $$\Phi(\boldsymbol{b}\_j) = \alpha_{1j} \boldsymbol{c}\_1 + \cdots + \alpha_{mj} \boldsymbol{c}\_m = \sum_{i=1}^m \alpha_{ij} \boldsymbol{c}\_i$$
> is the unique representation of $\Phi(\boldsymbol{b}\_j)$ with respect to $C$. Then we call the $m \times n$-matrix $\boldsymbol{A}\_{\Phi}$, whose elements are given by
> $$A_{\Phi} (i, j) = \alpha_{ij}$$
> the transformation matrix of $\Phi$.

즉, $V$의 각 기저에 대한 변환이 $W$의 기저 $C$로 유일하게 표현된다면 우리는 각 기저의 선형결합 계수들로 Transformation matrix를 만들 수 있다.$V$는 $\boldsymbol{b}_j$의 span이므로 $V$의 모든 기저에 대한 변환이 $W$공간에서 유일하게 표현된다면 $V$공간의 좌표에 대해 기저를 변환시키는 선형결합을 그대로 적용해주면 공간에 대한 변환이 되는 것도 직관적으로 이해해볼 수 있다. 이를 그대로 벡터에 대해 적용해보면 각 벡터에 대한 변환은 다음처럼 쓸 수 있다.

> If $\hat{\boldsymbol{x}}$ is the coordinate vector of $\boldsymbol{x} \in V$ with respect to $B$ and $\hat{\boldsymbol{y}}$ the coordinate vector of $\boldsymbol{y} = \Phi(\boldsymbol{x}) \in W$ with respect to C, then
> $$\hat{\boldsymbol{y}} = \boldsymbol{A}_{\Phi}\hat{\boldsymbol{x}}$$

다음의 선형사상을 보자.

$$
\boldsymbol{A} = 
\begin{bmatrix}
\cos \left(\frac{\pi}{4}\right) & -\sin \left(\frac{\pi}{4}\right) \\
\sin \left(\frac{\pi}{4}\right) & \cos \left(\frac{\pi}{4}\right)
\end{bmatrix}
$$

2차원 Euclidean의 basis는

$$
\boldsymbol{e}_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, \boldsymbol{e}_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}
$$

이다.

이 때 $\boldsymbol{e}_1$은 $\frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}$로, $\boldsymbol{e}_2$는 $\frac{1}{\sqrt{2}}\begin{bmatrix}-1 \\ 1\end{bmatrix}$ 각각 mapping 된다. 즉 $45^\circ$만큼 반시계방향으로 회전했음을 알 수 있다. 당연히 기저가 변하였으므로 이들의 span인 기존 공간의 좌표도 모두 같은 크기만큼 회전하게된다.