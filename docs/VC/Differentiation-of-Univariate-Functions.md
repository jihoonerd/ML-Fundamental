
# Differentiation of Univariate Functions

이번 포스팅은 단변량(univariate) 함수에 대한 미분으로 기초 미적분에 대한 내용과 동일하다.

## Difference Quotient

Difference quotient는 미소변화량에대한 함수의 변화량으로 다음과 같이 정의한다.

> [!NOTE]
> **Definition: Difference Quotient**
>
> $$ \frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x}$$
> Difference quotient는 함수 $f$에 대해서 미소변화량에 대한 기울기를 의미한다.

> [!NOTE]
> **Definition: Derivative**
>
> $h > 0 $에 대해서 함수 $f$의 $x$에서의 변화량은 다음과 같이 표현할 수 있다.
> $$\frac{df}{dx} := \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$$

## Taylor Series

Taylor Series는 어떤 함수 $f$를 $x_0$에서의 미분값의 합으로 표현하는 것으로 다항식(polynomial)으로 근사시키게 된다. 어떤 함수를 특정 값 기준으로 근사하는 도구로써 유용하게 사용할 수 있다.

> [!NOTE]
> **Definition: Taylor Polynomial**
>
> $n$차 Taylor polynomial로 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$를 $x_0$에서 정의하면 다음과 같다.
> $$ T_n(x) := \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^{k}$$
> $f^{(k)} (x_0)$는 $f$를 $k$번 미분한 함수의 $x_0$값에 해당한다. 그리고 $\frac{f^{(k)}(x_0)}{k!}$는 polynomial의 계수가 된다.

Taylor polynomial을 정의했으니 이제 Taylor series를 정의하자.

> [!NOTE]
> **Definition: Taylor Series**
>
> Smooth function $f \in \mathcal{C}^{\infty }, f: \mathbb{R} \rightarrow \mathbb{R}$에 대해서 함수 $f$의 $x_0$에서의 Taylor series는 다음과 같이 정의된다.
> $$T_{\infty } = \sum_{k=0}^{\infty } \frac{f^{(k)}(x_0)}{k!} (x - x_0)^{k}$$
> $x_0 = 0$일 때 Taylor series의 특수한 경우인 Maclaurin series가 된다. $f(x) = T_{\infty}(x)$일 때 $f$를 analytic하다고 부른다. 다시 말해, $x_0$ 근방에서 수렴하는 급수가 존재하면 해석함수라고 한다.

한편, Taylor series는 아래와 같이 polynomial의 합으로 나타내는 power series의 특수한 경우이다.
$$ f(x) = \sum_{k=0}^{\infty} a_k (x-c)^{k} $$

## Differentiation Rules

기본적인 미분법칙은 다음과 같다.

* Product Rule:
  $$ (f(x)g(x))^{\prime} = f^{\prime}(x)g(x) + f(x)g^{\prime}(x) $$
* Quotient rule:
  $$ \left( \frac{f(x)}{g(x)} \right)^{\prime} = \frac{f^{\prime}(x)g(x) - f(x)g^{\prime}(x)}{(g(x))^2} $$
* Sum rule:
  $$ (f(x) + g(x))^{\prime} = f^{\prime} (x) + g^{\prime} (x) $$
* Chain rule: 
  $$ (g(f(x)))^{\prime} = (g \circ f)^{\prime} (x) = g^{\prime} (f(x)) f^{\prime} (x) $$

## Conclusion

이번 문서에서는 기초적인 단변량(univariate) 미분에 대해서만 다루었다. 이어지는 단원에서는 편미분(paritial differentiation)과 gradient를 다루게 된다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.