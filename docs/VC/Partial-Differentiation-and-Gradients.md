# Partial Differentiation and Gradients

이번 문서 편미분(partial differentiation)과 gradient에 대해 다룬다. 앞에서는 단변량(univariate) 함수에 대해서만 다루었다면 여기서는 범위를 확장하여 한개 이상의 변수를 인자로 가지는 함수에 대해 다룬다. 그리고 다수의 변수에 대한 함수의 미분을 **gradient**라고 부른다. Gradient는 다른 변수를 고정시키고 한 변수에 대해서 하나씩 미분하여 구할 수 있다. 따라서 gradient는 partial derivative의 집합으로 볼 수 있다.

## Partial Derivative

> [!NOTE]
> **Definition: Partial Derivative**
>
> $n$개의 변수  $x_1, \ldots, x_n$에 대해서 $f$: $\mathbb{R}^{n} \rightarrow \mathbb{R}, \boldsymbol{x} \mapsto f(\boldsymbol{x}), \boldsymbol{x} \in \mathbb{R}^{n}$의 편미분은 다음과 같이 정의된다. 
> $$ \begin{aligned} \frac{\partial f}{\partial x_1} &= \lim_{h \rightarrow 0} \frac{f(x_1 + h, x_2, \ldots, x_n) - f(\boldsymbol{x})}{h} \\ &\vdots \\ \frac{\partial f}{\partial x_n} &= \lim_{h \rightarrow 0} \frac{f(x_1, x_2, \ldots, x_n + h) - f(\boldsymbol{x})}{h} \end{aligned} $$
> 이를 row vector로 모아주면 다음과 같다.
> $$ \begin{aligned} \nabla_x f = \text{grad} f &= \frac{df}{d \boldsymbol{x}} \\ &= \left[ \frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \cdots, \frac{\partial f(\boldsymbol{x})}{\partial x_n}  \right] \in \mathbb{R}^{1 \times n} \end{aligned} $$
> 이 row vector를 $f$의 **gradient**라고 한다.

## Basic ㄲules of Partial Differentiation

마찬가지로 편미분에서도 단변량에서 사용한 기본법칙들이 있으며 표현형은 단변량의 일반화된 형태로 생각하면 된다.

* Product rule:
  $$ \frac{\partial}{\partial \boldsymbol{x}} (f(\boldsymbol{x}) g(\boldsymbol{x})) = \frac{\partial f}{\partial \boldsymbol{x}} g(\boldsymbol{x}) + f(\boldsymbol{x}) \frac{\partial g}{\partial \boldsymbol{x}}$$
* Sum rule:
  $$ \frac{\partial}{\partial \boldsymbol{x}} (f(\boldsymbol{x}) + g(\boldsymbol{x})) = \frac{\partial f}{\partial \boldsymbol{x}} + \frac{\partial g}{\partial \boldsymbol{x}} $$
* Chain rule:
  $$ \frac{\partial}{\partial \boldsymbol{x}} (g \circ f)(\boldsymbol{x}) = \frac{\partial}{\partial \boldsymbol{x}}(g(f(\boldsymbol{x}))) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial \boldsymbol{x}} $$

여기서 Chain rule에 대해서 좀 더 자세히 살펴보자.

## Chain Rule

두 개의 변수 $x_1, x_2$를 인자로 갖는 함수 $f: \mathbb{R}^2 \rightarrow \mathbb{R}$가 있다고 해보자. 그리고 $x_1(t), x_2 (t)$는 모두 $t$의 함수이다. 이 때, 함수 $f$의 $t$에 대한 gradient는 Chain rule로 다음과 같이 풀어 쓸 수 있다.

$$ \frac{df}{dt} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} \begin{bmatrix} \frac{\partial x_1 (t)}{\partial t} \\ \frac{\partial x_2(t)}{\partial t} \end{bmatrix} = \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t} $$

만약 $x_1, x_2$가 $t$만의 함수가 아니라 $s, t$의 함수라면 어떻게 될까? 각 변수에 대한 편미분식으로 나타내기만 하면 된다.

$$
\begin{aligned}
\frac{\partial f}{\partial s} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial s} \\
\frac{\partial f}{\partial t} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t}
\end{aligned}
$$

이 gradient는 행렬의 곱으로 표현하면 다음처럼 나타낼 수 있다.

$$
\frac{df}{d(s, t)} = \frac{\partial f}{\partial \boldsymbol{x}} \frac{\partial \boldsymbol{x}}{\partial (s, t)} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} \begin{bmatrix} \frac{\partial x_1}{\partial s} & \frac{\partial x_1}{\partial t} \\ \frac{\partial x_2}{\partial s} & \frac{\partial x_2}{\partial t} \end{bmatrix}
$$

지금까지 모든 벡터는 column vector로써 정의하였는데 유독 책에서 gradient에 대해서만 row vector로 정의한다. 위의 식에서 이유가 드러나는데 행렬의 곱으로 깔끔하게 표현하는 방식은 오직 gradient가 row vector로써 표현되어야 저렇게 쓸 수 있기 때문이다. 하지만 이는 결국 표기의 차이일 뿐이므로 문맥에 따라 다른 곳에서는 얼마든지 column벡터로써 표현하기도 한다.

## Conclusion

편미분과 gradient에 대해서 다루었다. 특히, 미분규칙에서 편미분의 chain rule은 중요한 개념이므로 숙지하는 것이 좋다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.