# Matrix Approximation

수학적인 이론을 실제 application에 적용할 때, approximation은 유용한 도구이다. 특히, 컴퓨터로 이론적인 계산값을 완벽하게 재현할 수 있는 경우는 드물기 때문에 approximation을 필연적으로 해야하는 경우가 많다. 따라서 이론적 값과 상황에 따라 필요한 최소한의 오차 이내로 추정하는 것은 중요한 문제이다. 행렬은 그 자체로 변환하는 역할을 한다. 따라서 행렬을 더 낮은 Rank의 행렬의 합으로 표현할 수 있다면 계산상의 이점을 얻을 수 있을 것이다.

## Rank-$k$ approximation

SVD는 행렬 $\boldsymbol{A}$를 다음과 같이 factorize한다.

$$ \boldsymbol{A} = \boldsymbol{U \Sigma} \boldsymbol{V}^{\top} \in \mathbb{R}^{m \times n} $$

여기서는 SVD가 어떻게 행렬 $\boldsymbol{A}$를 더 낮은 랭크의 행렬의 합으로써 표현하는지를 살펴볼 것이다.

$\boldsymbol{U}, \boldsymbol{V}$ 각각의 $i$번째 column vector를 사용하면 rank-$1$ 행렬 $\boldsymbol{A}_{i} \in \mathbb{R}^{m \times n}$은 두 벡터의 곱으로 다음과 같이 분해할 수 있다.

$$
\boldsymbol{A} := \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{\top}
$$

Rank-$r$ 행렬 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$은 Rank-$1$ 의 합으로 다음과 같이 풀어쓸 수 있다.

$$
\boldsymbol{A} = \sum_{i=1}^{r} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{v}^{\top} = \sum_{i=1}^{r} \sigma_{i} \boldsymbol{A}_{i}
$$

참고로, $\boldsymbol{u} \boldsymbol{v}^{\top}$을 $\boldsymbol{u}, \boldsymbol{v}$의 outer-product라고 한다. 위의 식은 $\boldsymbol{A}_{i}$의 $i$번째 singular values의 가중합으로써 나타낸 것이라고 볼 수 있다. $i > r$인 Outer-product는 해당하는 singular value가 0이므로 0이 될 것이다.

위 식에서 rank-$r$인 행렬을 $k < r$인 column vector까지의 합으로 나타낸 것을 **rank-$k$ approximation**이라고 한다.

$$
\boldsymbol{\hat{A}}(k) := \sum_{i=1}^{k} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{\top} = \sum_{i=1}^{k} \sigma_{i} \boldsymbol{A}_{i}
$$

이 때, $\text{rk}(\boldsymbol{\hat{A}}(k)) = k$이다.

Rank-$k$ approximation을 이미지에 적용한 예를 살펴보자.

<figure align=center>
<img src="assets/images/LA/Fig_4.11.png" width=60% height=60%/>
<figcaption>Fig 4.11</figcaption>
</figure>

위 그림은 singular value 크기에 따라 차례대로 이미지를 나타낸 것이다. 첫 번째 행렬 $\boldsymbol{A}_{1}$은 그림의 전반적인 특징을, 그리고 이후의 행렬은 세부적인 특징을 잡아내고 있다. 이는 fourier series에서 low-frequency signal부터 high-frequency signal까지 더해가는 과정과 비슷하다. 위의 행렬들에 rank-$k$ approximation을 적용한 그림은 다음과 같이 된다.

<figure align=center>
<img src="assets/images/LA/Fig_4.12.png" width=60% height=60%/>
<figcaption>Fig 4.12</figcaption>
</figure>

Rank-$3$까지만 가도 스톤헨지의 형태를 식별할 수 있다. 원본이미지가 $1432 \times 1910 = 2,735,120$의 숫자로 표현된 것에 반해, rank-$5$ approximation은 5개의 singular value와 left/right singular vector만 사용한다. 즉 필요한 정보량은 $5 \cdot (1,432 + 1,910 + 1) = 16,715$뿐이다. 이는 원본 대비 0.6%의 숫자만으로 표현한 것임을 감안할 때 정보의 압축이 상당히 효율적으로 이루어졌음을 알 수 있다.

[여기부터]
# 2 Spectral Norm of a Matrix

그렇다면 원본이미지와 Approximation을 한 이미지의 차이는 어떻게 비교할 수 있을까? 거리에 대한 정의는 Norm으로 앞서 다루었다. 하지만 지금 비교해야 하는 대상은 벡터가 아닌 행렬이다. 이 때 사용할 수 있는 도구가 Sepctral Norm이다.

> **Definition: Spectral Norm of a Matrix**
>
> For $x \in \mathbb{R}^{n} \setminus \\{\boldsymbol{0}\\}$, the *spectral norm* of a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ is defined as:
> $$ \lVert \boldsymbol{A} \rVert_{2} := \max_{x} \frac{\lVert \boldsymbol{Ax} \rVert_{2}}{\lVert x \rVert_{2}} $$

Spectral norm은 임의의 벡터 $x$가 $\boldsymbol{A}$와 곱해졌을 때 가질 수 있는 최대 길이를 결정한다. 이러한 의미에 따라 다음 Theorem도 성립한다.

> **Theorem**
>
> The spectral norm of $\boldsymbol{A}$ is its largest singular value $\sigma_{1}$.


> **Theorem: Eckart-Young Theorem**
>
> Consider a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ of rank $r$ and let $\boldsymbol{B} \in \mathbb{R}^{m \times n}$ be a matrix of rank $k$. For any $k \leqslant r$ with $\boldsymbol{A}(k) = \sum_{i=1}^{k} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{\top} $ it holds that:
> $$\begin{eqnarray} \boldsymbol{\widehat{A}}(k) &=& \text{argmin}_{\text{rk}(\boldsymbol{B}) = k} \lVert \boldsymbol{A} - \boldsymbol{B} \rVert_{2} \\ \lVert \boldsymbol{A} - \boldsymbol{\widehat{A}}(k) \rVert_{2} &=& \sigma_{k+1} \end{eqnarray}$$

Eckart-Young Theorem은 $\boldsymbol{A}$를 근사하는 과정에서 얼마만큼의 오차가 생기는지를 보여주며 Spectral norm의 정의에 의해 $k$ rank까지 근사했다면 가질 수 있는 최대 오차는 $k+1$의 singular value $\sigma_{k+1}$임을 보여준다. 기하적으로는 rank-$k$ approximation은 Full-rank 행렬 $\boldsymbol{A}$를 최대 $k$차원으로 Projection한 것으로 해석할 수 있다. (Projection 포스팅에서 다루었던 최단거리의 개념에 대응한다) 즉, 가능한 모든 Projection 중에서 SVD는 $\boldsymbol{A}$와 rank-$k$ approximation이 최소한의 Spectral norm을 갖도록 하는 Projection을 찾는 것으로 볼 수 있다. 결과적으로 Eckart-Young Theorem은 우리가 SVD를 통해 임의의 행렬 $\boldsymbol{A}$를 더 낮은 차원의 랭크를 갖는 행렬로 근사시키는 체계적인 도구를 제공한다. 여기서 체계적이라 함은 Spectral norm을 최소화하는 방향으로 Optimization한다는 명확한 Objective function을 가짐을 의미한다. 물론, $k+1$이후의 Singular values로 대변되는 Column vectors에 대한 정보를 잃어버리게 되므로 Lossy compression의 일종이다.

# 3 Example Problem

앞서 SVD주제에서 다루었던 영화평가행렬에 대해 Approximation을 적용하는 예제를 살펴보자. 교재의 Example 4.15이다.

해당 문제는 SVD를 적용하면 다음과 같이 분해되었었다.

![Fig_4.10](/assets/images/2020-07-27-MML-04-05-Matrix-Decompositions/Fig_4.10.png){: .align-center}

여기서 Rank-$1$만을 사용해 Approximation을 하면 어떻게 될까?

$$
\begin{eqnarray}
\boldsymbol{A}_{1} &=& \boldsymbol{u}_{1} \boldsymbol{v}_{1}^{\top} \\
&=& \begin{bmatrix} -0.6710 \\ -0.7197 \\ -0.0939 \\ -0.1515 \end{bmatrix} \begin{bmatrix} -0.7367 & -0.6515 & -0.1811 \end{bmatrix} \\
&=& \begin{bmatrix} 0.4943 & 0.4372 & 0.1215 \\ 0.5302 & 0.4689 & 0.1303 \\ 0.0692 & 0.0612 & 0.0170 \\ 0.1116 & 0.0987 & 0.0274 \end{bmatrix}
\end{eqnarray}
$$

눈여겨 볼 점은 Rank-$1$ Approximation은 Ali와 Beatrix는 상대적으로 잘 살명하지만 Chandra에 대해서는 제대로 설명하지 못하고 있다. 이는 $\sigma_{1}$을 사용한 Approximation으로 해당 Singular value가 연결하는 것이 Sci-Fi 영화를 선호하는 사람과 영화관의 관계라는 점을 볼 때 예상가능한 결과이다. 같은 이유로 $\sigma_{2}$에 대응하는 Rank-$1$ Approximation은 Chandra와 같이 French Film쪽을 잘 설명하리라는 것을 예측할 수 있다. 실제로 그런지 값을 대입해보자.

$$
\begin{eqnarray}
\boldsymbol{A}_{2} &=& \boldsymbol{u}_{2} \boldsymbol{v}_{2}^{\top} \\
&=& \begin{bmatrix} 0.0236 \\ 0.2054 \\ -0.7705 \\ -0.6030 \end{bmatrix} \begin{bmatrix} 0.0852 & 0.1762 & -0.9807 \end{bmatrix} \\
&=& \begin{bmatrix} 0.0020 & 0.0042 & -0.0231 \\ 0.0175 & 0.0362 & -0.2014 \\ -0.0656 & -0.1358 & 0.7556 \\ -0.0514 & -0.1063 & 0.5914 \end{bmatrix}
\end{eqnarray}
$$

예상대로 $\sigma_{2}$에 대응하는 Rank-$1$ approximation은 Chandra의 평점을 잘 표현하는 반면, Sci-Fi쪽의 Approximation은 실제 값과 괴리가 큼을 볼 수 있다.

이제 이 둘의 Rank-$2$ Approximation을 계산해보자.

$$
\begin{eqnarray}
\boldsymbol{\widehat{A}}(2) &=& \sigma_{1} \boldsymbol{A}_{1} + \sigma_{2} \boldsymbol{A}_{2} \\
&=& \begin{bmatrix} 4.7801 & 4.2419 & 1.0244 \\ 5.2252 & 4.7522 & -0.0250 \\ 0.2493 & -0.2743 & 4.9724 \\ 0.7495 & 0.2756 & 4.0278 \end{bmatrix}
\end{eqnarray}
$$

이 결과는 원래의 $\boldsymbol{A}$를 매우 잘 근사시키고 있음을 보여주며 따라서 기존 행렬 $\boldsymbol{A}$라는 변환이 만들어 내는 movie-themes/movie-lovers 공간은 Sci-fi movies/lovers와 French art house movies/lovers의 Span으로 충분히 설명되고 있다고 할 수 있다.

# 4 Conclusion

이번 포스팅에서는 Matrix Approximation 방법에 대해 알아보았다. 아주 단순하게 요약하면, SVD결과에서 Singular values가 순서대로 각각 영향력이 큰 Rank-$1$의 공간을 설명하므로 이들을 순차적으로 더하면 기존 $\boldsymbol{A}$의 공간을 근사시키게 된다. 물론 효과적인 Approximation을 위해 $k$를 어떻게 설정해야할지는 문맥에 따라 다르겠지만 Eckart-Young Theorem은 합리적인 Baseline을 제공한다.

# 5 Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.