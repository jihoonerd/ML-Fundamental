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

**Marginal probability** $p(x)$는 random variable $Y$에 상관없이 $X=x$가 일어날 확률이다. 또한 random variable $X$가 확률분포 $p(x)$를 따르고 있따면 $X \sim p(x)$로 표현한다. $Y=y$일 때의 $X=x$의 확률을 conditional probability라고 하며 $p(y\mid x)$로 표현한다.


## Continuous PRobabilities