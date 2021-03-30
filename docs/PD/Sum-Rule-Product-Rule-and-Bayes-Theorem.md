# Sum Rule, Product Rule, and Bayes' Theorem

확률을 이용해 불확실성에 대해 다루려고 할 때 모든 과정은 sum rule과 product rule 두 토대 위에 만들어진다.

두 random variable $\boldsymbol{x}, \boldsymbol{y}$에 대해 joint distribution $p(\boldsymbol{x}, \boldsymbol{y})$가 있다고 해보자. 이 때, $p(\boldsymbol{x})$, $p(\boldsymbol{y})$를 각각 marginal distribution, $p(\boldsymbol{y} \mid \boldsymbol{x})$를 $\boldsymbol{y}$ given $\boldsymbol{x}$의 conditional distribution이라고 한다.

## Sum Rule

$$
p(\boldsymbol{x})=\left\{\begin{array}{ll}
\sum_{\boldsymbol{y} \in \mathcal{Y}} p(\boldsymbol{x}, \boldsymbol{y}) & \text { if } \boldsymbol{y} \text { is discrete } \\
\int_{\mathcal{Y}} p(\boldsymbol{x}, \boldsymbol{y}) \mathrm{d} \boldsymbol{y} & \text { if } \boldsymbol{y} \text { is continuous }
\end{array}\right.
$$

$\mathcal{Y}$는 random variable $Y$의 target space이다. 식에서 나타내 듯, $\boldsymbol{x}$의 marginalization은 가능한 모든 $\mathcal{Y}$를 summation하여 $\boldsymbol{y}$값에 상관없이 $p(\boldsymbol{x})$를 보겠다는 것이다. Sum rule은 marginalization property라고도 한다.

두 개 이상의 random variable이 있다면 여러 변수에 대해서 sum out하면 된다. 예를 들어 $\boldsymbol{x} = [x_{1}, \ldots, x_{D}]^{\top}$이라면 marginal은 다음과 같다.
$$p\left(x_{i}\right)=\int p\left(x_{1}, \ldots, x_{D}\right) \mathrm{d} \boldsymbol{x}_{\backslash i}$$

$\textbackslash i$는 "all except $i$"의 의미이다.

## Product Rule

Product rule은 joint distribution과 conditional distribution을 다음과 같이 연결해준다.

$$p(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{y} \mid \boldsymbol{x}) p(\boldsymbol{x})$$

Product rule은 joint distribution을 factorize하는 관점에서 바라볼 수도 있다. 위의 식은 joint distribution $p(\boldsymbol{x}, \boldsymbol{y})$를 conditional distribution $p(\boldsymbol{y} \mid \boldsymbol{x})$와 marginal distribution $p(\boldsymbol{x})$로 factorization으로 보면 된다.

## Bayes' Theorem

Bayes' theorem은 머신러닝과 확률 각각에서 중요한 의미를 가지며 probabilistic DL에서도 핵심을 이루는 개념이다. Bayes' theorem이 말하는 바는 posterior는 likelihood와 prior에 비례한다는 것이다.

> [!NOTE]
>
> **Bayes Theorem**
> $$p(\boldsymbol{x} \mid \boldsymbol{y}) = \frac{p(\boldsdymbol{y} \mid \boldsymbol{x}) p(\boldsymbol{x})}{p(\boldsymbol{y})}$$

어떤 확률분포에 대한 사전지식인 prior $p(\boldsymbol{x})$가 있고 관측하지 못한 데이터 $p(\boldsymbol{x})$가 다른 random variable $\boldsymbol{y}$와의 관계 $p(\boldsymbol{y} \mid \boldsymbol{x})$를 안다면 위의 Bayes' theorem을 이용해 $p(\boldsymbol{y})$를 보고 $p(\boldsymbol{x})$의 관계를 계산할 수 있다. 오른쪽 항의 분모에 해당하는 $p(\boldsymbol{y})$는 evidence라고 하는데 확률의 합이 1이 되도록 맞추어주는 역할을 한다. Evidence도 결국 marginalization을 통해 구하는 값인데 이 과정이 현실적으로 불가능하거나 어려운 경우가 많다. Bayes' theorem은 앞서 다룬 product rule을 통해 어렵지 않게 유도할 수 있다.




## Conclusion



## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
