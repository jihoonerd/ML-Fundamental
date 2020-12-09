# Determinant and Trace

이 문서에서는 determinant와 trace에 대해 다룬다. Determinant와 trace는 선형대수학에서 중요하게 사용되는 개념이다.

## Determinant 

> [!NOTE]
> **Definition: Determinant**
> 
> Determinant는 정사각행렬(Square matrix, $\boldsymbol{A} \in \mathbb{R}^{n \times n}$)에서만 정의되며 $\operatorname{det}(\boldsymbol{A})$ 또는 $\lvert \boldsymbol{A} \rvert$로 표기한다. 
> $$\operatorname{det}(\boldsymbol{A}) = \begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix}$$
> Determinant는 행렬 $\boldsymbol{A}$에서 실수 $\mathbb{R}$로의 함수이다.

Determinant는 어떤 정사각행렬 $\boldsymbol{A}$의 역행렬이 존재하는지를 간단하게 파악할 수 있게 해준다. 예를 들어, $2 \times 2$ 크기의 행렬 $\boldsymbol{A} = \left[ \begin{array}{cc} a_{11} & a_{12} \\ a_{21} & a_{22} \end{array} \right]$의 역행렬은 $\boldsymbol{A} \boldsymbol{A}^{-1} = \boldsymbol{I}$를 만족하므로 이 때의 $\boldsymbol{A}^{-1}$를 전개하면 다음과 같다.

$$
\boldsymbol{A}^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}}
\left[
\begin{array}{cc}
a_{22} & -a_{12} \\
-a_{21} & a_{11}
\end{array}\right]
$$

위로부터 자연스럽게 $\boldsymbol{A}$가 역행렬을 갖기 위해서는 $a_{11}a_{22} - a_{12}a_{21} \neq 0$을 만족해야 함을 알 수 있다. 이 때 분모부분 $a_{11}a_{22} - a_{12}a_{21}$을 $\boldsymbol{A} \in \mathbb{R}^{2 \times 2}$의 determinant라고 하며 다음과 같이 표현한다.

$$
\operatorname{det}(\boldsymbol{A})=\left|\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right|=a_{11} a_{22}-a_{12} a_{21}
$$

또한, 위의 성질은 $2 \times 2$에 국한되지 않고 일반적으로 성립하는 성질이다.

> [!NOTE]
> **Theorem**
> 
> 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$이 역행렬을 갖기 위해서는 $\operatorname{det}(\boldsymbol{A}) \neq 0$이어야 하며 역도 성립한다.

1~3차원 행렬의 determinant는 다음과 같다.

* $n=1$
  $$
  \begin{aligned}
  \operatorname{det}(\boldsymbol{A}) &= \operatorname{det}(a_{11}) \\
  &= a_{11}
  \end{aligned}
  $$

* $n=2$
  $$
  \begin{aligned}
  \operatorname{det}(\boldsymbol{A}) &=
  \begin{vmatrix}
  a_{11} & a_{12} \\
  a_{21} & a_{22}
  \end{vmatrix} \\
  &= a_{11}a_{22} - a_{12}a_{21}
  \end{aligned}
  $$

* $n=3$
  $$
  \begin{aligned}
  \operatorname{det}(\boldsymbol{A}) &= 
  \begin{vmatrix}
  a_{11} & a_{12} & a_{13} \\
  a_{21} & a_{22} & a_{23} \\
  a_{31} & a_{32} & a_{33}
  \end{vmatrix} \\
  &= a_{11}a_{22}a_{33} + a_{21}a_{32}a_{13} + a_{31}a_{12}a_{23}\\
  & - a_{31}a_{22}a_{13} - a_{11}a_{32}a_{23} -a_{21}a_{12}a_{33}
  \end{aligned} 
  $$

만약 행렬이 triangular matrix라면 대각성분을 곱하는 것만으로 Determinant를 계산할 수 있다.

$$
\operatorname{det}(\boldsymbol{T}) = \prod_{i=1}^{n} T_{ii}
$$

대수적으로는 위와 같은 의미를 갖고 계산되지만 determinant는 기하학적인 의미도 있다. 각 열벡터가 선형독립이라면 determinant는 열벡터가 표현하는 공간에 대해 2차원에서는 면적확장률, 3차원에서는 부피확장률을 의미하게 된다.

## Computing Determinant

Determinant를 구하는 일반식은 Laplace Exapnsion을 이용해 구할 수 있다.

> [!NOTE]
> **Theorem: Laplace Expansion**
>
> 행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$에 대해서 열, 또는 행을 $j = 1, \ldots, n$로 나타낼 때 다음이 성립한다.
> 1. Expansion along column $j$:
>   $$\operatorname{det}(\boldsymbol{A}) = \sum_{k=1}^{n} (-1)^{k+j}a_{kj}\operatorname{det}(\boldsymbol{A}_{k,j})$$
> 2. Expansion along row $j$:
>   $$\operatorname{det}(\boldsymbol{A}) = \sum_{k=1}^{n} (-1)^{k+j}a_{jk}\operatorname{det}(\boldsymbol{A}_{j,k})$$
> 
> 이 때, $\boldsymbol{A}_{k, j} \in \mathbb{R}^{(n-1) \times (n-1)}$은 $k$행과 $j$열을 지워서 얻을 수 있는 $\boldsymbol{A}$의 submatrix이다.

## Properties of Determinant

행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$의 determinant는 다음의 성질을 갖는다.

* $\operatorname{det}(\boldsymbol{AB}) = \operatorname{det}(\boldsymbol{A})\operatorname{det}(\boldsymbol{B})$
* $\operatorname{det}(\boldsymbol{A}) = \operatorname{det}(\boldsymbol{A}^\top)$
* $\boldsymbol{A}$가 역행렬을 가질 때 $\operatorname{det}(\boldsymbol{A}^{-1}) = \frac{1}{\operatorname{det}(\boldsymbol{A})}$이다.
* Similar matrices는 같은 determinant를 가진다. (행렬 $\boldsymbol{A}$와 행렬 $\boldsymbol{B}$가 non-singular matrix $\boldsymbol{S}$를 가지고 $\boldsymbol{A} = \boldsymbol{S}^{-1}\boldsymbol{BS}$를 만족할 때 행렬 $\boldsymbol{A}, \boldsymbol{B}$는 similar하다고 한다. Similar관계에 있는 두 행렬은 Basis는 다르지만 같은 Linear mapping이다.)
* 한 행/열의 배수를 다른 행/열에 더하는 것은 $\operatorname{det}(\boldsymbol{A})$를 바꾸지 않는다.
* $\operatorname{det}(\lambda\boldsymbol{A}) = \lambda^n \operatorname{det}(\boldsymbol{A})$
* 행/열의 교환(swapping)은 $\operatorname{det}(\boldsymbol{A})$의 부호를 바꾼다.

Determinant는 rank와 관련해서 다음의 정리가 성립한다.

> [!NOTE]
> **Theorem**
>
> $\boldsymbol{A}$가 역행렬을 갖는 것은 $\boldsymbol{A}$가 full rank인 것과 필요충분조건의 관계를 갖는다.

## Trace

> [!NOTE]
> **Definition: Trace**
>
> 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$의 trace는 다음과 같이 정의한다.
> $$\operatorname{tr}(\boldsymbol{A}) := \sum_{i=1}^n a_{ii}$$
> 즉, trace는 대각성분의 합이다.

## Properties of Trace

Trace는 다음의 성질을 갖는다.

* $\operatorname{tr}(\boldsymbol{A} + \boldsymbol{B}) = \operatorname{tr}(\boldsymbol{A}) + \operatorname{tr}(\boldsymbol{B}) \quad \operatorname{for} \ \boldsymbol{A, B} \in \mathbb{R}^{n \times n}$
* $\operatorname{tr}(\alpha\boldsymbol{A}) = \alpha \operatorname{tr}(\boldsymbol{A}), \alpha \in \mathbb{R} \quad \operatorname{for} \ \boldsymbol{A} \in \mathbb{R}^{n \times n}$
* $\operatorname{tr}(\boldsymbol{I}_n) = n$
* $\operatorname{tr}(\boldsymbol{AB}) = \operatorname{tr}(\boldsymbol{BA}) \quad \operatorname{for} \ \boldsymbol{A} \in \mathbb{R}^{n \times k}, \boldsymbol{B} \in \mathbb{R}^{k \times n}$
* Trace는 cyclic permutation에 대해서 불변이다. $(\operatorname{tr}(\boldsymbol{AKL}) = \operatorname{tr}(\boldsymbol{KLA}))$ 여기서 재미있는 성질을 유도할 수 있는데 similar matrix의 trace가 같다는 것은 cyclic permuation 불변성질로 간단히 보일 수 있다. $(\operatorname{\boldsymbol{B}} = \operatorname{tr}(\boldsymbol{S}^{-1} \boldsymbol{A} \boldsymbol{S}) = \operatorname{tr}(\boldsymbol{AS}\boldsymbol{S}^{-1}) = \operatorname{tr}(\boldsymbol{A}))$

## Determinant and Trace from Characteristic Polynomial

다음과 같이 행렬을 다항식으로 표현하는 characteristic polynomial의 계수로 determinant와 trace를 구할 수 있다.

> [!NOTE]
> **Definition: Characteristic Polynomial**
>
> $\lambda \in \mathbb{R}$와 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$에 대해서 $c_0, \ldots, c_{n-1} \in \mathbb{R}$인 다음의 표현식 $p_{\boldsymbol{A}}(\lambda)$를 **characteristic polynomial**이라고 한다:
> $$\begin{aligned}
  p_{\boldsymbol{A}}(\lambda) :&= \operatorname{det}(\boldsymbol{A} - \lambda \boldsymbol{I}) \\
  &= c_{0} + c_{1}\lambda + c_{2}\lambda^2 + \cdots + c_{n-1}\lambda^{n-1} + (-1)^n \lambda^n
  \end{aligned}$$
> 이 때, determinant와 trace는 각각 상수항과 $\lambda$의 $n-1$차항의 계수에 대응된다.
> $$c_{0} = \operatorname{det}(\boldsymbol{A})$$
> $$c_{n-1} = (-1)^{n-1}\operatorname{tr}(\boldsymbol{A})$$

Characteristic polynomial은 다음에 다룰 Eigenvalues와 Eigenvectors를 구할 때 사용된다.

# 3 Conclusion

교재에서 determinant와 trace는 행렬의 분해(matrix decomposition)을 다루기에 앞서 소개되는 개념이다. 행렬분해를 다루기 위해서 필요하다는 것을 의미한다. 각각은 반드시 알아야하는 개념이며 아무래도 determinant가 rank와의 관계나 기하학적인 의미 등 다양한 의미를 지니다보니 시험이나 면접에서는 trace보다 더 자주 언급되는 편이다. 마지막에 소개된 characteristic polynomial은 특히 손계산으로 행렬분해와 관련된 문제를 풀 때 편리하게 사용할 수 있는 식이므로 익숙해지도록 하자.

# 4 Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
