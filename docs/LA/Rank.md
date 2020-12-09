# Rank

행렬의 성질을 확인할 때 가장 자주 확인하는 것 중 하나가 rank일 것이다. 특히, 어떤 행렬이 full rank인지 아닌지에 따라서 선형시스템의 해나 행렬의 변환특성은 크게 달라진다. 이 문서에서는 rank에 대해 알아보자.

## Rank

Rank는 다음가 같이 정의할 수 있다.

> [!NOTE]
> **Definition: Rank**
>
> 행렬 $\boldsymbol{A} \in \mathbf{R}^{m \times n}$에 대해 선형독립인 열벡터의 수는 선형독립은 행벡터의 수와 같고 이를 행렬 $\boldsymbol{A}$의 **rank** 라고한다. rank는 $\text{rk}(\boldsymbol{A})$로 표기한다.

## Properties of Rank

Rank가 가지는 성질은 다음가 같다.

* $\text{rk}(\boldsymbol{A}) = \text{rk}(\boldsymbol{A}^{\top})$: 행벡터, 열벡터의 rank는 같다.
* $\boldsymbol{A} \in \mathbf{R}^{m \times n}$의 열은 부분공간 $U \subseteq \mathbf{R}^{m}$을 span하며 $\text{dim}(U) = \text{rk}(\boldsymbol{A})$이다. 나중에 다루지만 이 부분공간을 **image** EHsms **range**라고 한다. $U$의 기저는 가우스 소거법을 통해 pivot columns를 확인함으로써 구할 수 있다.
* $\boldsymbol{A} \in \mathbf{R}^{n \times n}$의 행은 부분공간 $W \subseteq \mathbf{R}^{m}$을 span하며 $\text{dim}(W) = \text{rk}(\boldsymbol{A})$이다. $W$의 기저는 $\boldsymbol{A}^{\top}$에 대한 가우스 소거법을 통해 구할 수 있다.
* Regular (invertible) 행렬 $\boldsymbol{A} \in \mathbf{R}^{n \times n}$은 $\text{rk}(\boldsymbol{A})=n$이다.
* $\boldsymbol{A} \in \mathbf{R}^{m \times n}, \boldsymbol{b} \in \mathbf{R}^{m}$일때, 선형시스템 $\boldsymbol{Ax} = \boldsymbol{b}$는 $\text{rk}(\boldsymbol{A}) = \text{rk}(\boldsymbol{A} \mid \boldsymbol{b})$일때만 풀 수 있다.
* 행렬 $\boldsymbol{A} \in \mathbf{R}^{m \times n}$에 대해서 $\boldsymbol{Ax} = \boldsymbol{0}$의 해에 대한 부분공간은 $n - \text{rk}(\boldsymbol{A})$를 갖는다. 행렬 $\boldsymbol{A}$가 full rank라면 0차원으로 특정 해를 갖는다. 이러한 부분공간을 **kernel** 또는 **null space** 라고한다.
* 행렬 $\boldsymbol{A} \in \mathbf{R}^{m \times n}$이 가능한 최대의 rank를 가질 때, 즉 행렬츼 차원과 동일한 rank를 가질 때 **full rank** 라고 한다. 따라서 full rank일 경우 $\text{rk}(\boldsymbol{A}) = \text{min}(m, n)$이 성립한다. Full-rank가 아닐경우 **rank deficient** 라고 한다.
* 행렬 $\boldsymbol{A} \in \mathbf{R}^{n \times n}$일때, 역행렬이 존재하기 위해서는 반드시 full rank여야 한다.

## Example

1. 다음 행렬의 rank를 구해보자.

  $$
  \boldsymbol{A}=\begin{bmatrix}
    1 & 0 & 1 \\
    0 & 1 & 1 \\
    0 & 0 & 0
  \end{bmatrix}
  $$

  두 개의 pivot column이 있으므로 $\text{rk}(\boldsymbol{A})$는 2이다.

2. 다음 행렬의 rank를 구해보자.

  $$
  \boldsymbol{A}=\begin{bmatrix}
    1 & 2 & 1 \\
    -2 & -3 & 1 \\
    3 & 5 & 0
  \end{bmatrix}
  $$

  위 행렬에 대해 가우스 소거법을 사용하면 $\begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 3 \\ 0 & 0 & 0 \end{bmatrix}$이다. 위와 마찬가지로 두 개의 pivot column을 가지므로 $\text{rk}(\boldsymbol{A})=2$이다.

## Conclusion

여기서는 rank의 정의와 가우스 소거법을 기준으로 rank를 구하는 법을 보았지만 rank는 다양한 방식으로 정의할 수 있다. 특히, 이후에 다룰 eigenvalue를 사용하면 특성방정식의 해로 rank를 구할 수도 있다. 면접에서도 단골주제이다. 선형대수학의 다른 개념, 특히 행렬분해와 관련된 내용은 면접시 손으로 풀기에는 번거로운 측면이 있으나 rank정도는 그 자리에서 바로 푸는 것이 충분히 가능하기 때문에 유독 자주 등장하는 것 같다. 필자의 경우도 지원했던 인공지능대학원 중 한 곳에서 선형대수학을 선택했었는데 rank를 구하는 문제가 나왔었다. 이후 개념을 위해서도, 면접대비로서도 반드시 알아야 하는 개념이다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
