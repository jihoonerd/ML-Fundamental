# Linearization and Multivariate Taylor Series

## Multivariate Taylor Series

어떤 함수 $f$의 gradient $\nabla f$를 안다면 특정 지점 $\boldsymbol{x}_{0}$ 근방에서 선형 근사를 할 수 있다.
$$
f(\boldsymbol{x}) \approx f\left(\boldsymbol{x}_{0}\right)+\left(\nabla_{\boldsymbol{x}} f\right)\left(\boldsymbol{x}_{0}\right)\left(\boldsymbol{x}-\boldsymbol{x}_{0}\right)
$$

<figure align=center>
<img src="assets/images/VC/Fig_5.12.png" width=60% height=60%/>
<figcaption>Fig 5.12</figcaption>
</figure>

특정지점에서의 선형근사이므로 $\boldsymbol{x}_{0}$에서 멀어질수록 이 근사의 오차는 커지게 된다. 더 큰 관점에서 보면 선형근사는 Taylor series의 특수한 경우라고 볼 수 있다. Taylor series가 무한급수로 표현하는 반면 선형근사는 Taylor series의 앞의 두개 항만 사용한다고 볼 수 있다. 따라서 항을 추가할수록 더 넓은 범위에서의 근사도 정확해질 것이다.

Taylor series에 의한 근사법은 간단한 원리이지만 실제로도 많이 사용되는 방법이다. 여기서는 다변수(multivariate) Taylor series에 대해 다룬다.

> [!NOTE]
> **Definition: Multivariate Taylor Series**
>
> 다음과 같이 입력으로 $D$의 크기를 갖는 벡터를 받아 실수를 출력하는 함수를 생각해보자.
> $$\begin{aligned}f: \mathbb{R}^{D} & \rightarrow \mathbb{R} \\ \boldsymbol{x} & \mapsto f(\boldsymbol{x}), \quad \boldsymbol{x} \in \mathbb{R}^{D} \end{aligned}$$
> 이 함수가 $\boldsymbol{x}_{0}$에서 smooth하다면 $\boldsymbol{x}_{0}$ 근방의 벡터 $\boldsymbol{x}$에 대해 차이를 $\boldsymbol{\delta} \coloneqq \boldsymbol{x} - \boldsymbol{x}_{0}$라고 정의하자. 이 때 $f$의 $\boldsymbol{x}_{0}$에서의  multivariate Taylor series는 다음과 같이 정의할 수 있다.
> $$f(\boldsymbol{x})=\sum_{k=0}^{\infty} \frac{D_{\boldsymbol{x}}^{k} f\left(\boldsymbol{x}_{0}\right)}{k !} \boldsymbol{\delta}^{k}$$
> $D_{\boldsymbol{x}}^{k} f(\boldsymbol{x}_{0})$는 $f$를 $\boldsymbol{x}$에 대해서 $k$번 미분한 함수의 $\boldsymbol{x}_{0}$에서의 값이다.

> [!NOTE]
> **Taylor Polynomial**
> 
> 함수 $f$의 $\boldsymbol{x}_{0}$에서의 degree $n$ Taylor polynomial은 $n+1$개의 항을 가지며 다음과 같이 표현된다.
> $$T_{n}(\boldsymbol{x})=\sum_{k=0}^{n} \frac{D_{\boldsymbol{x}}^{k} f\left(\boldsymbol{x}_{0}\right)}{k !} \boldsymbol{\delta}^{k}$$
> 
이 때 $D_{\boldsymbol{x}}^{k}f$와 $\boldsymbol{\delta}^{k}$의 차원을 신경써야 하는데 만약 $\boldsymbol{x} \in \mathbb{R}^{D}$로 $\boldsymbol{x}$가 $D>1$차원의 벡터이고 $k>1$이라면 $k$-th order tensor $\boldsymbol{\delta}^{k}$는 다음과 같이 $k$의 outer product로 얻어진다.
$$\boldsymbol{\delta}^{k} \in \mathbb{R}^{\overbrace{D \times D \times \ldots \times D}^{k \text { times }}}$$
외적을 $\otimes$라고 정의하면 다음과 같이 표기할 수 있다.
$$\boldsymbol{\delta}^{2}:=\boldsymbol{\delta} \otimes \boldsymbol{\delta}=\boldsymbol{\delta} \boldsymbol{\delta}^{\top}, \quad \boldsymbol{\delta}^{2}[i, j]=\delta[i] \delta[j]$$
$$\boldsymbol{\delta}^{3}:=\boldsymbol{\delta} \otimes \boldsymbol{\delta} \otimes \boldsymbol{\delta}, \quad \boldsymbol{\delta}^{3}[i, j, k]=\delta[i] \delta[j] \delta[k]$$

이를 일반화하면 다음과 같다.
$$D_{x}^{k} f\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta}^{k}=\sum_{i_{1}=1}^{D} \cdots \sum_{i_{k}=1}^{D} D_{\boldsymbol{x}}^{k} f\left(\boldsymbol{x}_{0}\right)\left[i_{1}, \ldots, i_{k}\right] \delta\left[i_{1}\right] \cdots \delta\left[i_{k}\right]$$

$k=0,\ldots,3$인 경우에 대해서는 다음과 같다. 차원에 유의하자.
$$\begin{aligned}
k=0&: D_{x}^{0} f\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta}^{0}=f\left(\boldsymbol{x}_{0}\right) \in \mathbb{R} \\
k=1&: D_{\boldsymbol{x}}^{1} f\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta}^{1}=\underbrace{\nabla_{\boldsymbol{x}} f\left(\boldsymbol{x}_{0}\right)}_{1 \times D} \underbrace{\boldsymbol{\delta}}_{D \times 1}=\sum_{i=1}^{D} \nabla_{\boldsymbol{x}} f\left(\boldsymbol{x}_{0}\right)[i] \delta[i] \in \mathbb{R} \\
k=2&: D_{\boldsymbol{x}}^{2} f\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta}^{2}=\operatorname{tr}(\underbrace{\boldsymbol{H}\left(\boldsymbol{x}_{0}\right)}_{D \times D} \underbrace{\boldsymbol{\delta}}_{D \times 1} \underbrace{\boldsymbol{\delta}^{\top}}_{1 \times D})=\boldsymbol{\delta}^{\top} \boldsymbol{H}\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta} \\
&=\sum_{i=1}^{D} \sum_{j=1}^{D} H[i, j] \delta[i] \delta[j] \in \mathbb{R}\\
k=3&: D_{\boldsymbol{x}}^{3} f\left(\boldsymbol{x}_{0}\right) \boldsymbol{\delta}^{3}=\sum_{i=1}^{D} \sum_{j=1}^{D} \sum_{k=1}^{D} D_{x}^{3} f\left(\boldsymbol{x}_{0}\right)[i, j, k] \delta[i] \delta[j] \delta[k] \in \mathbb{R}
\end{aligned}$$

## Conclusion

이번 문서에서는 다변량(multivariate) Taylor series에 대해 알아보았다. 단변량(univariate)과 마찬가지이나 벡터로 입력이 들어가는 만큼 표현이 복잡해진다. 개인적으로 이 부분은 손계산 할 일은 없고 원리 자체는 단변량일 때와 같은 개념을 확장한 것이므로 벡터인 경우의 Taylor series는 이렇게 된다는 정도로만 받아들이고 가도 괜찮다고 생각한다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.