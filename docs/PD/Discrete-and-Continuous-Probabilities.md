# Discrete and Continuous Probabilities

Target space가 연속값을 갖는지, 이산값을 갖는지에 따라 분포를 표현하는 방식이 달라지게 된다. 

Target space $\mathcal{T}$가 이산값을 갖는다면 random variable $X$의 확률값을 특정 값에 대해 정의할 수 있다.
$$P(X=x)$$
이산값을 갖는 random variable $X$에 대한 $P(X=x)$를 **probability mass function**이라고 한다.

Target space $\mathcal{T}$가 연속값을 갖는다면 random variable $X$의 확률은 interval에 의해서 정의된다.
$$P(a \leqslant X \leqslant b) \quad \text{ for } a < b$$
그리고 $P(X \leqslant x)$에 대해서 cumulative distribution function이라고 한다.

## Discrete Probabilities

Target space가 discrete하다면 확률분포의 표현은 표를 채우는 형태로 생각해볼 수 있다. Random variable $X, Y$에 대한 joint probability는 다음과 같이 정의할 수 있다.

$$P(X=x_{i}, Y=y_{j}) = \frac{n_{ij}}{N}$$

$n_{ij}$는 $x_{i}, y_{i}$가 발생한 횟수이며 $N$은 모든 event의 횟수이다. Joint probability는 두 event의 교집합으로 다음과 같이 표현이 가능하다.
$$P\left(X=x_{i}, Y=y_{j}\right)=P\left(X=x_{i} \cap Y=y_{j}\right)$$
이러한 discrete한 random variable의 확률 분포를 **probability mass function**이라고 한다. 그리고 $p(x,y)$를 joint probability라고 한다. 이를 함수처럼 $x, y$를 인자로 받아 실수인 확률을 반환하는 함수로 바라볼 수도 있다.

**Marginal probability** $p(x)$는 random variable $Y$에 상관없이 $X=x$가 일어날 확률이다. 또한 random variable $X$가 확률분포 $p(x)$를 따르고 있따면 $X \sim p(x)$로 표현한다. $Y=y$일 때의 $X=x$의 확률을 **conditional probability**라고 하며 $p(y\mid x)$로 표현한다.

## Continuous Probabilities

Target space가 실수 $\mathbb{R}$일 때의 continuous probability에 대해 알아보자. 엄밀한 정의를 위해서는 집합의 크기인 measure와 Borel $\sigma$-algebra까지 다루어야 하지만 여기서는 연속확률 자체보다는 성질과 활용을 다루므로 책에서 다루는 실수 random variable은 Borel $\sigma$ algebra에 해당한다는 내용정도만 받아들이고 이후 내용을 다루어보자.

> [!NOTE]
>
> **Definition: Probability Density Function** 다음을 만족하는 $f: \mathbb{R}^{D} \rightarrow \mathbb{D}$를 pdf라고 한다.
> * $\forall \boldsymbol{x} \in \mathbb{R}^{D} : f(\boldsymbol{x}) \geqslant 0$
> * $\int_{\mathbb{R}^{D}} f(\boldsymbol{x}) d \boldsymbol{x} = 1$

pdf는 양수를 함수값으로 가지며 전체구간의 적분이 1이되는 함수임을 알 수 있다. $a, b \in \mathbb{R}, x \in \mathbb{R}$이고 $X$가 연속인 random variable이면 다음이 성립한다.
$$P(a \leqslant X \leqslant b) = \int_{a}^{b} f(x) dx$$
Discrete random variable과는 다르게 특정한 random variable $P(X=x)$은 0이다.

> [!NOTE]
> 
> **Definition: Cumulative Distribution Function**: real-valued random valriable $X$에 대해서 각 상태가 $D$ 차원으로 $\boldsymbol{x} \in \mathbb{R}^{D}$일때, cumulative distribution function은 다음과 같다.
> $$F_{X}(\boldsymbol{x}) = P(X_{1} \leqslant x_{1}, \ldots, x_{D} \leqslant x_{D})$$

cdf는 적분으로 표현하면 아래와 같이 된다.
$$F_{X}(\boldsymbol{x})=\int_{-\infty}^{x_{1}} \cdots \int_{-\infty}^{x_{D}} f\left(z_{1}, \ldots, z_{D}\right) \mathrm{d} z_{1} \cdots \mathrm{d} z_{D}$$

## Contrasting Discrete and Continuous Distributions

확률은 양수의 값을 가지며 총 합은 1이 되어야 한다. 따라서 discrete random variable은 각각의 state가 $[0, 1]$에 있지만 continuous random variable은 전체적분값이 1이되면 될 뿐 pdf는 1보다 큰 값을 가질 수도 있다.

<figure align=center>
<img src="assets/images/PD/Fig_6.3.png" width=100% height=100%/>
<figcaption>Fig 6.3</figcaption>
</figure>

교재에서 discrete과 continuous random variable에 대해서 깔끔하게 표로 정리해주고 있다.

|Type|Point Probability|Interval Probability|
|---|---|---|
|Discrete|$P(X=x)$: pmf|Not Applicable|
|Continous|$p(x)$: pdf|$P(X\leqslant x)$: cdf|

## Conclusion

Discrete probability와 continuous probability에 대해 알아보았다.엄밀하게는 분명 구분되는 개념이고 명명법도 따로 있지만 저자가 언급하듯, 머신러닝에서는 이 둘을 기술적으로 아주 정확하게 사용하지는 않고 문맥에 어느정도 의존하고 있다. 이해하는데 큰 문제가 없는 선에서 용어가 다소 혼재할 수 있음을 인지하고 교재를 보면 좋다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
