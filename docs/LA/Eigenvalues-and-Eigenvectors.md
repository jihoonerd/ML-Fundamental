# Eigenvalues and Eigenvectors

이번 문서에서는 eigenvalue와 eigenvector에 대해 다룬다. 행렬분해에 있어 가장 기본이 되는 개념이며 행렬의 변환을 기하학적으로 생각할 때, 공간에 대해 이해할 수 있는 토대를 제공해준다. 면접에서도 선형대수학 부분에서는 최우선순위에 있는 주제 중 하나이다.

## Eigenvalue and Eigenvector

선형변환은 basis로 특정되는 고유의 변환 행렬로 나타낼 수 있다. 이러한 선형변환은 "eigen" analysis를 통해 해석할 수 있다. Eigenvalue와 eigenvector의 정의를 알아보자.

> [!NOTE]
> **Definition: Eigenvalue, Eigenvector, Eigenvalue equation**
>
> 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$에 대해서 다음의 식을 만족할 때, $\lambda \in \mathbb{R}$를 $\boldsymbol{A}$의 **eigenvalue**라고 하며 $\boldsymbol{x} \in \mathbb{R}^n \setminus \{\boldsymbol{0}\}$를 eigenvalue $\lambda$에 대한 $\boldsymbol{A}$의 **eigenvector**라고 한다.
> $$\boldsymbol{Ax} = \lambda \boldsymbol{x}$$
> 그리고 위 식을 **eigenvalue equation**이라고 한다.

Eigenvalue는 sorting이 되어있는 경우 분석에 용이하나 software사용시 꼭 보장되는 성질은 아니다. 예로 Python의 Numpy는 eigenvalue 기준으로 sorting이 되어있지는 않다.([numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)) 교재에서도 별도로 언급이 되지 않는한, sorting이 되어있지 않다고 가정한다.

Eigenvalue와 eigenvector 정의에 의해 다음은 모두 같은 의미를 갖는다.

* $\lambda$는 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$의 eigenvalue이다.
* $\boldsymbol{Ax} = \lambda \boldsymbol{x}$를 만족하는 $\boldsymbol{x} \in \mathbb{R}^{n} \setminus \{\boldsymbol{0}\}$가 존재한다.
* $(\boldsymbol{A} - \lambda \boldsymbol{I}_{n}) \boldsymbol{x} = \boldsymbol{0}$의 해가 non-trivial solution이다. $(\boldsymbol{x} \neq \boldsymbol{0})$
* $\operatorname{rk}(\boldsymbol{A} - \lambda\boldsymbol{I}_{n}) < n$
  Full-rank가 아니다는 이야기이다. 밑의 성질을 확인하고 다시 보면 이해하기가 쉽다.
* $\operatorname{det}(\boldsymbol{A} - \lambda\boldsymbol{I}_{n}) = 0$
  Characteristic polynomial의 해가 존재한다는 의미이다. 그리고 그 해가 eigenvalue이다.

> [!NOTE]
> **Definition: Collinearity and Codirection**
> 
> 두 vector가 같은 방향을 가리키는 방향이 같다면 **codirected**라고 한다. 두 벡터가 한 직선위에 있다면, 다시 말해 같은 방향 혹은 반대방향을 가리킨다면 **collinear**라고 한다.

어떤 eigenvalue에 대응하는 eigenvector가 있을 때, 이 eigenvector의 상수배는 모두 eigenvector이다. 즉 eigenvalue에 대해 eigenvector는 unique하지 않다. Vector $\boldsymbol{x}$가 $\boldsymbol{A}$의 eigenvector라면 $\boldsymbol{x}$와 collinear한 벡터도 모두 $\boldsymbol{A}$의 eigenvector이다.

Eigenvalue는 위에서 언급한 것처럼 characteristic polynomial을 풀어서 얻을 수 있다. 다음의 정리가 성립한다.

> [!NOTE]
> **Theorem: Eigenvalue and the Root of the Characteristic Polynomial**
>
> Eigenvalue는 characteristic polynomial의 해이다. 즉, $\boldsymbol{A} \in \mathbb{R}^{n \times n}$의 eigenvalue $\lambda \in \mathbb{R}$는 $\boldsymbol{A}$의 **characteristic polynomial** $p_{\boldsymbol{A}}(\lambda)$의 해를 통해 구할 수 있다.

다음으로 algebraic multiplicity와 geometric multiplicity에 대해 알아보자.

> [!NOTE]
> **Definition: Algebraic Multiplicity and Geometric Multiplicity**
>
> 정사각행렬 $\boldsymbol{A}$의 eigenvalue $\lambda_{i}$에 대해서 characteristic polynomial에서 구할 수 있는 해의 개수를 **algebraic multiplicity**라고 하며, 선형독립인 eigenvector들의 개수를 **geometric multiplicity**라고 한다. Geometric multiplicity는 해당 $\lambda$의 eigenvector가 span한 eigenspace의 차원으로 볼 수 있다. Geometric multiplicity는 algebraic multiplicity보다 클수는 없고 같거나 적을 수(중근을 가질 때)는 있다. 그러나 1보다는 항상 크다.

Eigenvalue, eigenvector에 비해 생소하긴 하지만 eigenspace와 eigenspectrum의 정의도 알아보자. 간단한 내용이다.

> [!NOTE]
> **Definition: Eigenspace, Eigenspectrum**
>
> 행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$에 대해서 $\boldsymbol{A}$의 eigenvector가 span하는 $\mathbb{R}^{n}$차원 부분공간을 eigenvalue $\lambda$에 대한 $\boldsymbol{A}$의 **eigenspace**라고 하며 $E_{\lambda}$라고 한다. 그리고 $\boldsymbol{A}$의 모든 eigenvalue의 집합을 **eigenspectrum**이라고 한다.

Eigenvalue와 eigenvector를 정의하는 식인 $\boldsymbol{Av} = \lambda\boldsymbol{v}$를 생각해보면, 각각의 eigenvalue $\lambda$에 대한 eigenspace $E_{\lambda}$는 선형방정식 $(\boldsymbol{A}-\lambda\boldsymbol{I})\boldsymbol{v} = \boldsymbol{0}$의 solution space이다. 기하학적으로는 linear mapping $\boldsymbol{Av}$는 Eigenvector에 대해 $\lambda$만큼 늘리거나 줄인 것으로 볼 수 있다.

이 내용은 eigenvalue, eigenvector의 정의와 연계지어서 꼭 이해하여야 하는 개념 중 하나이다. 특히, 기하학적으로 벡터에 대한 행렬의 곱은 해당 벡터에 대해 linear mapping을 적용하는 것이고 그 결과가 동일한 벡터의 상수배라는 결과는 벡터공간에서 eigenvector가 어떤 벡터인지를 보다 잘 이해하게 해준다. 즉, 특정 행렬의 eigenvector에 대한 linear mapping은 방향을 바꾸지 않는다.

실제 대학원 면접에서 질문받았던 내용 중 하나로 eigenvalue와 eigenvector의 정의가 있었다. 당연히 정의식 $\boldsymbol{Av} = \lambda \boldsymbol{v}$를 기준으로 설명하는 것이 좋지만 이렇게 끝내기에는 매우 밋밋한다. 따라서 eigenvalue와 eigenvector의 대수적, 기하학적 의미정도를 부연설명하는 것은 공부측면에서도 면접관에게 이해하고 있다라는 걸 전달하기 위해서도 좋다고 생각한다.

## Useful Properties of Eigenvalue and Eigenvector

Eigenvalue와 eigenvector는 다음의 성질을 갖는다.

* 행렬 $\boldsymbol{A}$와 Transpose $\boldsymbol{A}^\top$은 같은 eigenvalue를 가진다. 하지만 반드시 같은 eigenvector를 갖는 것은 아니다.
* Eigenspace $E_{\lambda}$는 $\boldsymbol{A} - \lambda \boldsymbol{I}$의 null space이다.
  $$\begin{aligned}
  \boldsymbol{Ax} = \lambda\boldsymbol{x} &\iff \boldsymbol{Ax} - \lambda\boldsymbol{x} = \boldsymbol{0}\\
  &\iff (\boldsymbol{A}-\lambda\boldsymbol{I})\boldsymbol{x} = \boldsymbol{0}\\
  &\iff \boldsymbol{x} \in \text{ker}(\boldsymbol{A} - \lambda \boldsymbol{I})
  \end{aligned}
  $$
* Similar matrices는 같은 eigenvalue를 가진다. 즉, eigenvalue를 비롯해 determinant, trace는 basis change에 대해 불변(invariant)이다.
* Symmetric, positive definite matrices는 항상 양의 실수인 eigenvalue를 갖는다.

## Example: Computing Eigenvalues, Eigenvectors, and Eigenspaces

Eigenvalues, Eigenvectors, Eigenspaces를 구하는 방법을 예제문제로 알아보자. 순서는 아래와 같이 접근하면 된다.

1. Characteristic polynomial
2. Eigenvalue
3. Eigenvector and Eigenspace

### Example 1

다음의 행렬에 이를 적용해 보자.

$$
\boldsymbol{A} = \begin{bmatrix}4 & 2 \\ 1 & 3\end{bmatrix}
$$

#### Characteristic polynomial

Eigenvector는 영벡터가 아니다. 따라서 $(\boldsymbol{A}- \lambda \boldsymbol{I})\boldsymbol{x} = 0$에서 $\boldsymbol{x} \neq 0$이면 이 선형방정식은 영벡터 이외의 해를 가져야 한다. 즉 영벡터라는 유일한 해로써 정의되지 않으며 $\boldsymbol{A} - \lambda \boldsymbol{I}$는 non-invertible해야하므로 determinant는 0이 된다.

따라서, characteristic polynomial은 다음과 같다.

$$
p_{\boldsymbol{A}} = \text{det}(\boldsymbol{A} - \lambda \boldsymbol{I})
$$

#### Eigenvalue

Characteristic polynomial의 해를 구한다.

$$
\begin{aligned}
p_{\boldsymbol{A}} &= \text{det}(\boldsymbol{A} - \lambda \boldsymbol{I})\\
&= \text{det}\left(\begin{bmatrix}4 & 2 \\ 1 & 3\end{bmatrix} - \begin{bmatrix}\lambda & 0 \\ 0 & \lambda \end{bmatrix}\right)\\
&= \begin{vmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda\end{vmatrix}\\
&= (4-\lambda)(3-\lambda) - 2 \cdot 1
\end{aligned}
$$

Determinant가 0이될때의 조건을 찾고 있으므로 2차방정식을 풀면 된다.

$$
\begin{aligned}
p_{\boldsymbol{A}} &= (4-\lambda)(3-\lambda) - 2 \cdot 1\\
&= \lambda^2 -7\lambda + 10 \\
&= (\lambda-2)(\lambda-5) = 0
\end{aligned}
$$

따라서 eigenvalue는 각각 $\lambda_1 = 2, \lambda_2 = 5$이다.

#### Eigenvectors and Eigenspace

이제 eigenvalue를 알고 있으므로 정의 식에 대입하여 eigenvector를 구할 수 있다.

$$
\begin{bmatrix}
4-\lambda & 2 \\ 1 & 3-\lambda
\end{bmatrix} \boldsymbol{x} = \boldsymbol{0}
$$

1) $\lambda=5$
   
  위에 대입하면 다음과 같다.
  $$\begin{bmatrix}-1 & 2 \\ 1 & -2\end{bmatrix}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix} = \boldsymbol{0}$$
  정리하면 $x_1 = 2x_2$의 관계를 얻는다. 따라서 $\lambda=5$의 solution space는 다음과 같다.
  $$E_{5} = \text{span}[\begin{bmatrix}2 \\ 1 \end{bmatrix}]$$

2) $\lambda=2$
  
  위에 대입하면 다음과 같다.
  $$\begin{bmatrix}2 & 2 \\ 1 & 1\end{bmatrix}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix} = \boldsymbol{0}$$
  정리하면 $x_1 = -x_2$의 관계를 얻는다. 따라서 $\lambda=2$의 solution space는 다음과 같다.
  $$E_{2} = \text{span}[\begin{bmatrix}1 \\ -1 \end{bmatrix}]$$

이 경우는 algebraic multiplicity와 geometric multiplicity가 모두 2인 상황이다. 하지만 앞서 언급한대로 geometric multiplicity는 algebraic multiplicity보다 적을 수도 있다. 이 경우를 살펴보자.

### Example 2

다음 행렬에 대해 eigenvalue와 eigenvector를 구해보자.

$$
\boldsymbol{A} = \begin{bmatrix}2 & 1 \\ 0 & 2\end{bmatrix}
$$

위의 과정을 똑같이 적용하면 characteristic polynomial은 중근을 가지게되어 eigenvalue는 $\lambda_1 = \lambda_2 = 2$이다. 대수적 해는 2개이므로 algebraic multiplicity는 2이다. 하지만 eigenvector는 $\boldsymbol{x}_{1} = \begin{bmatrix}1 \\ 0 \end{bmatrix}$로 eigenspace는 1개의 eigenvector의 span으로만 구성된다. 따라서 geometric multiplicity는 1이다.

## Graphical Intuition in Two Dimensions

교재는 Fig 4.4에서 다섯가지 형태의 선형 변환에 대해 eigenvalue와 eigenvector가 어떻게 달라지는지 보여준다. 시각적으로 이해할 수 있게 도와주는 매우 유용한 자료이므로 꼭 살펴보도록 하자.

<figure align=center>
<img src="assets/images/LA/Fig_4.4.png"/>
<figcaption>Figure 4.4: Determinants and eigenspaces. Overview of five linear mappings and their associated transformation matrices.</figcaption>
</figure>

각 변환은 위에서부터 순서대로 $\boldsymbol{A}_1, \ldots, \boldsymbol{A}_5$에 해당한다.

* $\boldsymbol{A}_{1} = \begin{bmatrix}\frac{1}{2} & 0 \\ 0 & 2\end{bmatrix}$
  
  2차원 euclidean basis에서 $x$축은 절반으로 줄어들고, $y$축은 2배로 늘어난 형태이다. Determinant는 1로 면적확장률은 그대로이다. 따라서 정사각형의 영역은 $y$축으로 늘어당긴 모양의 직사각형이 된다.

* $\boldsymbol{A}_{2} = \begin{bmatrix}1 & \frac{1}{2} \\ 0 & 1\end{bmatrix}$
  
  Eigenvalue를 계산해보면 $\lambda_1 = 1 = \lambda_2$이다. 따라서 geometric multiplicity는 1이다. 이 때, $(\boldsymbol{A} - \lambda \boldsymbol{I})\boldsymbol{x}=\boldsymbol{0}$에 대입하면 $x_{2}=0$의 조건만을 얻게된다. 즉, Eigenvector는 $[c, 0]^\top$의 꼴로 $x$축으로만 변형을 시킨다. 첫번째 경우와 마찬가지로 determinant는 1이므로 면적확장률은 1로 기존 정사각형의 면적이 보존된다.

* $\boldsymbol{A}_{3} = \begin{bmatrix} \cos(\pi/6) & -\sin(\pi/6)  \\ \sin(\pi/6) & \cos(\pi/6) \end{bmatrix}$
  
  이 행렬은 $30^{\circ}$를 회전시키는 유명한 변환이다. 지금은 변환 그 자체도 중요하지만 eigenvalue나 eigenvector측면에서 바라보는 것이 중요하므로 eigenvalue를 풀어보면 다음과 같은 두 Conjugate value가 나온다. $\lambda_1 = (0.87 -0.5j), \lambda_2 = (0.87 + 0.5j)$ 흥미로운 점은 회전변환의 경우 복소수가 나타난다는 점이다. 이는 phase와 관련된 내용이나 주 관심사는 복소공간이 아니므로 이런 형태로 표현된다는 것만 알고 넘어가도 충분하다. 회전변환이 면적을 바꾸는 것은 아니므로 마찬가지로 Determinant는 1인것을 확인할 수 있다.

* $\boldsymbol{A}_{4} = \begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}$
  
  Characteristic polynomial은 $(1-\lambda)^2 - 1 = 0$으로 $\lambda_1 = 0, \lambda_2 = 2$이다. $\lambda_{1} = 0$의 eigenvector는 $x_1 = x_2$를 만족할 때이므로 $\begin{bmatrix}1 \\ 1 \end{bmatrix}$이다. 이 식을 Eigenvector/value정의식에 대입하면 다음과 같다.
  $$\boldsymbol{Ax} = 0 \cdot \begin{bmatrix}1 \\ 1 \end{bmatrix} $$
  해당 eigenvector성분, 즉 $\begin{bmatrix}1 \\ 1 \end{bmatrix}$방향성분이 0으로 줄어드는 것을 알 수 있다. 반면, $\lambda=2$일때의 eigenvector는 $\begin{bmatrix}1 \\ -1 \end{bmatrix}$로 해당방향 성분은 2배만큼 늘어난다. Determinant를 계산하면 0이되는데 면적확장률이 0, 즉 면적은 0으로 된다는 것을 확인할 수 있고 모든 점들이 선 위로 mapping되므로 위 그림의 결과와도 일치한다.

* $\boldsymbol{A}_{5} = \begin{bmatrix}1 & \frac{1}{2} \\ \frac{1}{2} & 1\end{bmatrix}$
  
  Characteristic polynomial을 계산하면 eigenvalue는 $\lambda_1 = 0.5, \lambda_2 = 1.5$이며 각각의 eigenvector는 $\begin{bmatrix}1 \\ -1 \end{bmatrix}$, $\begin{bmatrix}1 \\ 1 \end{bmatrix}$이다. 따라서 $\lambda_{1}$의 eigenvector성분은 절반 줄어들고, $\lambda_{2}$의 eigenvector성분은 1.5배로 늘어나는 것을 볼 수 있다. Determinant는 0.75인데 기존 면적에서 $\frac{1}{2} \cdot \frac{3}{2}$배만큼 변한 것이므로 일치한다.

## Spectral Theorem

앞으로 유요하게 사용할 정리인 spectral theorem을 다루기에 앞서 몇 가지 중요한 정리를 살펴보자.

> [!NOTE]
> **Theorem**
> 
> $\boldsymbol{A} \in \mathbb{R}^{n \times n}$가 $n$개의 서로 다른 eigenvalues $\lambda_1, \ldots, \lambda_n$를 갖는다면, 이 때의 eigenvector $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_{n}$는 선형독립이다. 그리고 $n$개의 eigenvector는 $\mathbb{R}^n$인 basis를 구성하게 된다.

> [!NOTE]
> **Definition: Defective Matrix**
> 
> 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$이 $n$개 미만의 선형독립인 eigenvector를 가질 경우 *defective*하다고 한다.

--------------[여기부터]

하지만 non-defective matrix가 반드시 $n$개의 서로다른 eigenvalue를 가질 필요는 없다. 다만 Non-defective하다면 $\mathbb{R}^{n}$의 basis를 구성해야 한다. 따라서 defective matrix에서는 eigenspace의 차원수 합이 $n$보다 작다.

이제는 Symmetric과 Positive definite 성질에 집중해보자. 우선 다음의 정리를 보자.

> **Theorem**
>
> Given a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, we can always obtain a symmetric, positive semidefinite matrix $\boldsymbol{S} \in \mathbb{R}^{n \times n}$ by defining:
> $$\boldsymbol{S} := \boldsymbol{A}^\top \boldsymbol{A}$$
> If $\text{rk}(\boldsymbol{A}) = n$, then $\boldsymbol{S} := \boldsymbol{A}^\top \boldsymbol{A}$ is symmetric, positive definite.

각 성질의 성립여부는 위의 $\boldsymbol{S}$를 대입하면 쉽게 얻을 수 있다. 실제로, $\boldsymbol{S} = \boldsymbol{S}^\top$을 만족하며 positive semidefiniteness인 $\boldsymbol{x}^\top \boldsymbol{Sx} \geq 0$를 만족한다. 벡터/행렬로 표현되었을 뿐 스칼라의 제곱에 대응하는 개념이다.

### Example 3 (4.8)

다음 행렬의 Eigenvector를 구해보자.

$$
\boldsymbol{A} = \begin{bmatrix}
3 & 2 & 2 \\
2 & 3 & 2 \\
2 & 2 & 3 \\
\end{bmatrix}
$$

Characteristic polynomial은 다음과 같이 얻어진다.

$$
p_{\boldsymbol{A}}(\lambda) = -(\lambda-1)^2 (\lambda-7)
$$

1) $\lambda_1 = 1$
   
  Characteristic polynomial이 중근을 갖는 경우이다. 이 때 eigenvector는 두개가 얻어지며 eigenspace는 다음과 같다.
  $$E_{1} = \text{span}[\begin{bmatrix}-1 \\ 1 \\ 0\end{bmatrix}, \begin{bmatrix}-1 \\ 0 \\ 1\end{bmatrix}]$$

2) $\lambda_2 = 7$

  Eigenspace는 다음과 같다.
  $$E_{7} = \text{span}[\begin{bmatrix}1 \\ 1 \\ 1\end{bmatrix}]$$

  $E_{7}$이 $E_{1}$의 두 벡터와 직교함에 유의하자. 또한 위 행렬은 symmetric matrix로 잠시뒤에 다룰 spectral theorem에 의해 orthonormal basis로 벡터공간을 구성할 수 있다. Gram-Schmidt process를 $\boldsymbol{x}_1, \boldsymbol{x}_3$에 적용하면 $E_{1}$은 다음과 같이 Orthonormal basis로 바꿔 쓸 수 있다.

$$E_{1} = \text{span}[\begin{bmatrix} -1 \\ 1 \\ 0\end{bmatrix}, \frac{1}{2} \begin{bmatrix} -1 \\ -1 \\ 2\end{bmatrix}]$$


## 4 Spectral Theorem

Spectral theorem은 이후 다룰 Eigendecomposition이 Symmetric 행렬에 대해 *항상* 존재하고 Diagonalization이 됨을 뒷받침한다. Spectral theorem은 다음과 같다.

> **Theorem: Spectral theorem**
>
> If $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ is symmetric, there exists an orthonormal basis of the corresponding vector space $V$ consisting of eigenvectors of $\boldsymbol{A}$, and each eigenvalue is real.

## 5 Eigenvalue/vector and Determinant/Trace

Eigenvalue/vector와 Determinant/Trace간의 유용한 관계를 정리하자.

> **Theorem**
>
> The determinant of a matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ is the product of its eigenvalues, i.e.,
> $$\text{det}(\boldsymbol{A}) = \prod_{i=1}^n \lambda_{i}$$
> where $\lambda_{i} \in \mathbb{C}$ are (possibly repeated) eigenvalues of $\boldsymbol{A}$

> **Theorem**
>
> The trace of a matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ is the sum of its eigenvalues, i.e.,
> $$\text{tr}(\boldsymbol{A}) = \sum_{i=1}^{n} \lambda_i$$
> where $\lambda_{i} \in \mathbb{C}$ are (possibly repeated) eigenvalues of $\boldsymbol{A}$

## 6 Google's PageRank

초창기 구글이 서비스를 시작했을 무렵, 다른 경쟁자 대비 강력한 구글의 검색능력의 원천은 [PageRank](https://en.wikipedia.org/wiki/PageRank)라는 알고리즘에서 비롯되었다. (Page도 Larry Page에서 따온것이라고 한다) 검색엔진은 각 웹페이지의 중요도를 파악해 검색시 중요한 페이지를 우선적으로 보여주는 것이 중요하다. 뜬금없이 Eigenvalue/vector를 이야기하다가 갑자기 이게 무슨 이야기인지 싶지만 구글의 검색엔진이 작동하는 기본원리가 바로 Eigenvalue/vector에 있다.

이 알고리즘은 어떤 웹페이지의 중요도는 해당 페이지로의 링크에 의해 추정될 수 있다고 생각한다. 이를 표현하기 위해 PageRank는 Directed graph로 각 페이지가 어느 페이지로 링크되어 있는지를 표현한다.

![PageRank](https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/PageRanks-Example.jpg/600px-PageRanks-Example.jpg){: .align-center}

PageRank 알고리즘은 어떤 웹페이지 $a_{i}$가 얼마나 많이 다른 페이지로부터 링크되어있는지를 통해  가중치를 결정한다. 이 때 링크의 수이므로 가중치는 0보다 크거나 같은 값을 갖는다. 또한 $a_{i}$로 링크가 걸린 사이트를 셀 뿐만 아니라 각 페이지 내에서 가리키는 페이지들이 얼마나 있는지를 감안해 $a_{i}$의 가중치에 반영한다. 이 정보를 통해 Transition matrix $\boldsymbol{A}$로 모델링하여 확률을 계산하게된다. 이 Transition matrix는 초기값 vector $\boldsymbol{x}$에 대해서 연속적으로 변환하면, $\boldsymbol{x}, \boldsymbol{Ax}, \boldsymbol{A}^2 \boldsymbol{x}, \ldots$ 결과적으로 어떤 벡터 $\boldsymbol{x}^{\*}$로 수렴하는 성질이 있다. 이 수렴하는 벡터가 *PageRank*이며 $\boldsymbol{A}\boldsymbol{x}^{\*} = \boldsymbol{x}^{\*}$이 성립한다. 이 식을 통해 PageRank는 Eigenvalue가 1일때의 Eigenvector로 볼 수 있다. 그리고 PageRank를 normalize한 $\boldsymbol{x}^{\*}$를 통해 특정 웹페이지로 이동할 확률로 해석할 수 있다.

## 7 Conclusion

이번 장에서는 행렬분해뿐만 아니라 선형대수학에서 가장 자주 언급되는 개념인 Eigenvalue와 Eigenvector를 다루었다. 행렬분해에서는 이 개념을 반복해서 사용하게되므로 정의식뿐만 아니라 선형변환에서 Eigenvalue와 Eigenvector의 의미, 계산하는 방법정도는 충분히 암기할만한 가치가 있다.

## 8 Reference

* [Wikipedia: PageRank](https://en.wikipedia.org/wiki/PageRank)
* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.