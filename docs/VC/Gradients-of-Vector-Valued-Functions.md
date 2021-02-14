# Gradients of Vector-Valued Functions


지금까지 다룬 미분은 단변량(Univariate)이든 편미분이든 결국 미분하는 함수는 하나의 실수값을 치역(range)으로 갖는 함수였다. 하지만 머신러닝 모형이 분류문제를 푸는 것을 생각해보면 입력값으로 벡터로 된 정보가 들어오고 출력값은 분류 클래스에 대한 확률과 같은 벡터의 형태로 나오게 된다. 즉, 벡터에서 실수로의 mapping이 아닌 벡터에서 벡터의 mapping을 하는 vector-valued function에 대한 미분을 해야한다. 이번 포스팅에서는 이러한 함수에 대한 미분에 대해 다룬다.

## Jacobian

하나의 실수를 출력하는 함수는 다음과 같은 domain과 range를 갖는다.

$$ f: \mathbb{R}^{n} \rightarrow \mathbb{R} $$

반면, Vector-valued function은 다음과 같이 출력되는 차원이 스칼라 값이 아니다.

$$ \boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}, \quad \text{where} \ n \geqslant 1 \ \text{and} \ m > 1 $$

입력값으로 벡터 $\boldsymbol{x} = [x_{1}, \ldots, x_{n}]^{\top} \in \mathbb{R}^{n}$가 들어가면 함수값은 다음처럼 표현된다.

$$
\boldsymbol{f}(\boldsymbol{x}) = \begin{bmatrix} f_1(\boldsymbol{x}) \\ \vdots \\ f_{m}(\boldsymbol{x})  \end{bmatrix} \in \mathbb{R}^{m}
$$

위의 표기에서 $\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$을 풀어 쓰면 함수의 벡터 $[f_1, \ldots, f_m]^{\top}$로 나타낼 수 있다. 각각의 함수 $f_i: \mathbb{R}^{n} \rightarrow \mathbb{R}$는 다시 실수 range를 갖는 함수가 된다. 편미분의 내용을 이에 그대로 적용하면 다음과 같다.

$$
\frac{\partial \boldsymbol{f}}{\partial x_{i}}=\left[\begin{array}{c}
\frac{\partial f_{1}}{\partial x_{i}} \\
\vdots \\
\frac{\partial f_{m}}{\partial x_{i}}
\end{array}\right]=\left[\begin{array}{c}
\lim _{h \rightarrow 0} \frac{f_{1}\left(x_{1}, \ldots, x_{i-1}, x_{i}+h, x_{i+1}, \ldots x_{n}\right)-f_{1}(\boldsymbol{x})}{h} \\
\vdots \\
\lim _{h \rightarrow 0} \frac{f_{m}\left(x_{1}, \ldots, x_{i-1}, x_{i}+h, x_{i+1}, \ldots x_{n}\right)-f_{m}(\boldsymbol{x})}{h}
\end{array}\right] \in \mathbb{R}^{m}
$$

$\boldsymbol{f}$의 gradient는 row벡터가 됨을 편미분에서 다루었다. 우선 함수 $\boldsymbol{f}$는 신경쓰지 말고 $\boldsymbol{x}$에 대한 미분으로만 바라보면 row vector로 각 성분에 대한 미분으로 표기할 수 있다. 그리고 함수 $\boldsymbol{f}$를 $m$개의 output을 갖는 함수로 column방향으로 펼치면 다음과 같이 된다.

$$
\begin{aligned}
\frac{\mathrm{d} \boldsymbol{f}(\boldsymbol{x})}{\mathrm{d} \boldsymbol{x}} &=\left[\begin{array}{ccc}
\frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial x_{n}}
\end{array}\right] \\
&=\left[\begin{array}{ccc}
\frac{\partial f_{1}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{1}(\boldsymbol{x})}{\partial x_{n}} \\
\vdots & & \vdots \\
\frac{\partial f_{m}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{m}(\boldsymbol{x})}{\partial x_{n}}
\end{array}\right] \in \mathbb{R}^{m \times n}
\end{aligned}
$$

## Jacobian

> [!NOTE]
> **Definition: Jacobian**
> 앞서 다룬 gradient에 대한 정의를 상기해보면, $\boldsymbol{f}$의 gradient는 편미분값의 row vector로 정의되었었다. 따라서 vector-valued function $\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$의 $\boldsymbol{x} \in \mathbb{R}^{n}$에 대한 미분은 다음과 같이 표현할 수 있다.
> $$\begin{aligned}
\boldsymbol{J} = \nabla_{\boldsymbol{x}} \boldsymbol{f} = \frac{d \boldsymbol{f}(\boldsymbol{x})}{d \boldsymbol{x}} &= \begin{bmatrix} \frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial x_{n}} \end{bmatrix} \\
&= \begin{bmatrix}
\frac{\partial f_{1}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{1}(\boldsymbol{x})}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{m}(\boldsymbol{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{m}(\boldsymbol{x})}{\partial x_{n}}
\end{bmatrix} \in \mathbb{R}^{m \times n}
\end{aligned}
$$

이 때 $\boldsymbol{x} = [x_1, \cdots, x_n]^{\top}$, $J(i, j) = \frac{\partial f_i}{\partial x_j}$이다.
이러한 vector-valued function $\boldsymbol{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$의 1차 편미분을 **Jacobian**이라고 한다. Jacobian matrix의 원소 $(i, j)$는 $i$번째 함수 $f_{i}$의 $x_{j}$에 대한 미분이다.

$$ J(i, j) = \frac{\partial f_{i}}{\partial x_{j}} $$

여기서 주의할 점이 있는데 Jacobian과 같이 벡터에 대한 미분을 표기하는 방식은 **numerator layout**과 **denominator layout**이 있다. Numerator layout은 위의 표기와 같은 방식으로 gradient를 쓸 때, 함수벡터는 행을 확장하는 방향으로, $\boldsymbol{x}$는 열을 확장하는 방향으로 표기하는 방식이다. Denominator layout은 반대로 함수벡터가 열을 확장하는 방향으로 써지며 $\boldsymbol{x}$가 행을 확장하는 방향으로 쓰게 된다.

Jacobian은 change-of-variable 기법에서도 중요하게 활용된다. $\boldsymbol{b}$의 벡터공간에서 $\boldsymbol{c}$의 벡터공간으로의 mapping이 있다고 할 때 이 변환과정은 각각의 basis의 변화에 대해 Jacobian으로 표현할 수 있다. (Jacobian은 input도 벡터 output도 벡터임을 상기하자) 변화를 나타내는 Jacobian은 행렬이고 이 변환의 input과 output 차원이 같아서 정사각행렬이라면 determinant를 계산해볼 수 있다. Determinant를 계산하면 determinant가 갖는 공간확장율의 의미가 부여되어 mapping에 의한 공간확장율을 얻을 수 있게 된다.

인공지능 학습의 많은 부분은 derivative-based이며 이는 gradient, Jacobian, Hessian과 같은 식들로 표현이 되므로 이 개념들에 대해서는 직관적으로 떠올릴 수 있을만큼 익숙해져야 한다.


# 3 Example

교재에서는 $\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{Ax}$, Chain rule, Linear Model에 대한 미분을 예제로 보여준다.

## 3.1 Gradient of a Vector-Valued Function

$$ \boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{A} \boldsymbol{x}, \boldsymbol{f}(\boldsymbol{x}) \in \mathbb{R}^{M}, \boldsymbol{A} \in \mathbb{R}^{M \times N}, \boldsymbol{x} \in \mathbb{R}^{N} $$

위와 같이 주어졌을 때 gradient $ \frac{d \boldsymbol{f}}{d \boldsymbol{x}} $를 구해보자. 우선 차원을 먼저 확인해야 한다. $\boldsymbol{f}$를 $\boldsymbol{x}$로 미분하면 $\boldsymbol{f}$의 차원만큼 행이 생길 것이고 $\boldsymbol{x}$의 차원만큼 열이 생기므로 $ d \boldsymbol{f} / d \boldsymbol{x} \in \mathbb{R}^{M \times N} $으로 $M \times N$차원이 된다. 이제 각각의 미분값을 계산하면 되는데 각 행은 $f_{i}$에 의해 다음으로 계산할 수 있다.

$$
f_{i}(\boldsymbol{x}) = \sum_{j=1}^{N} A_{ij}x_{j} \Longrightarrow \frac{\partial f_{i}}{\partial x_{j}} = A_{ij}
$$

위 식은 $i$행에 적용된다. 따라서 위의 결과를 이용하면  $ \frac{d \boldsymbol{f}}{d \boldsymbol{x}} $는 다음과 같이 표현할 수 있다.

$$
\begin{aligned}
\frac{d \boldsymbol{f}}{d \boldsymbol{x}} &= 
\begin{bmatrix}
\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{1}}{\partial x_{N}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{M}}{\partial x_{1}} & \cdots & \frac{\partial f_{M}}{\partial x_{N}}
\end{bmatrix} \\
&=
\begin{bmatrix}
A_{11} & \cdots & A_{1N} \\
\vdots & \ddots & \vdots \\
A_{M1} & \cdots & A_{MN}
\end{bmatrix} \\
&= \boldsymbol{A} \in \mathbb{R}^{M \times N}
\end{aligned}
$$

## 3.2 Chain Rule

실수에서 실수로의 함수 $h: \mathbb{R} \rightarrow \mathbb{R}$은 합성함수로 $h(t) = (f \circ g)(t) $이고 다음과 같이 주어졌다.

$$
\begin{aligned}
f &: \mathbb{R}^{2} \rightarrow \mathbb{R} \\
g &: \mathbb{R} \rightarrow \mathbb{R}^{2} \\
f(\boldsymbol{x}) &= \exp({x_{1} x_{2}^{2}}) \\
\boldsymbol{x} &= \begin{bmatrix} x_{1} \\ x_{2} \end{bmatrix} = g(t) = \begin{bmatrix} t \cos t \\ t \sin t \end{bmatrix}
\end{aligned}
$$

이 때 $ \frac{dh}{dt} $를 구해보자.

앞의 예제와 마찬가지로 차원부터 확인하면,

$$ \frac{\partial f}{\partial \boldsymbol{x}} \in \mathbb{R}^{1 \times 2}, \frac{\partial g}{\partial t} \in \mathbb{R}^{2 \times 1}$$

$g(t) = \boldsymbol{x} $임을 상기하자. 이후에는 Chain Rule을 적용하고 대입해 계산만 하면 된다.

$$
\begin{aligned}
\frac{dh}{dt} &= \frac{\partial f}{\partial \boldsymbol{x}} \frac{\partial \boldsymbol{x}}{\partial t} \\
&= \begin{bmatrix} \frac{\partial f}{\partial x_{1}} & \frac{\partial f}{\partial x_{2}} \end{bmatrix} \begin{bmatrix} \frac{\partial x_{1}}{\partial t} \\ \frac{\partial x_{2}}{\partial t} \end{bmatrix}
\end{aligned}
$$

## 3.3 Gradient of a Least-Squares Loss in a Linear Model

여기서는 행렬 미분을 이용해 Least-Square방식을 적용한다.

다음과 같은 선형모형이 있다고 해보자.

$$
\boldsymbol{y} = \boldsymbol{\Phi \theta}
$$

여기서 $ \boldsymbol{\theta} \in \mathbb{R}^{D} $이며 $ \boldsymbol{\Phi} \in \mathbb{R}^{N \times D} $이다.

손실함수 $L$은 제곱오차 $ \Vert \boldsymbol{e} \Vert^{2} $로 정의하며 $\boldsymbol{e}$는 관측한 $ \boldsymbol{y} \in \mathbb{R}^{D} $와 현재 parameter에 의한 예측값인 $\boldsymbol{\hat{y}} = \boldsymbol{\Phi \theta}$의 차이로 정의한다.

$$
\begin{aligned}
L(\boldsymbol{e}) &:= \Vert \boldsymbol{e} \Vert^{2} \\
\boldsymbol{e}(\boldsymbol{\theta}) &:= \boldsymbol{y} - \boldsymbol{\Phi \theta}
\end{aligned}
$$

손실함수를 감소시키도록 $\boldsymbol{\theta}$를 업데이트해야하므로 $\frac{\partial L}{\partial \boldsymbol{\theta}}$를 구하면 된다. 두 식은 $\boldsymbol{e}$를 매개변수로 사용하므로,

$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial \boldsymbol{e}} \frac{\partial \boldsymbol{e}}{\partial \boldsymbol{\theta}}
$$

로 표현이 가능하다.

교재에서 $[1, D]$의 원소에 대해 이야기하는데 이 문제에서 $L$은 하나의 함수이다. ($\boldsymbol{L}$이 아니다!). 따라서 $[1, D]$의 원소라는 것은 $ \frac{\partial L}{\partial \theta_{D}} $이다.

$$
\frac{\partial L}{\partial \boldsymbol{\theta}}[1, d] = \sum_{n=1}^{N} \frac{\partial L}{\partial \boldsymbol{e}}[n] \frac{\partial \boldsymbol{e}}{\partial \boldsymbol{\theta}}[n, d]
$$

$ \frac{\partial L}{\partial \boldsymbol{e}}, \frac{\partial \boldsymbol{e}}{\partial \boldsymbol{\theta}} $는 다음처럼 계산되는데 이러한 미분공식은 교재 5.5를 참조하면 된다.

$$
\begin{aligned}
\frac{\partial L}{\partial \boldsymbol{e}} = 2 \boldsymbol{e}^{\top} \in \mathbb{R}^{1 \times N} \\
\frac{\partial \boldsymbol{e}}{\partial \boldsymbol{\theta}} = - \boldsymbol{\Phi} \in \mathbb{R}^{N \times D}
\end{aligned}
$$

위의 결과를 대입하면 다음과 같이 목표로 한 $\frac{\partial L}{\partial \boldsymbol{\theta}}$를 계산하면 된다.

$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = -2\boldsymbol{e}^{\top} \boldsymbol{\Phi} = -2(\boldsymbol{y}^{\top} - \boldsymbol{\theta}^{\top} \boldsymbol{\Phi}^{\top}) \boldsymbol{\Phi} \in \mathbb{R}^{1 \times D}
$$

굳이 합성함수로 풀지 않고 $L$을 바로 풀어버리는 것도 가능하며 교재 5.5의 미분공식을 이용하면 쉽게 얻을 수 있다. 

하지만 컴퓨터로 계산하는 환경에서는 Chain Rule이 압도적으로 유리하다. 해석적으로 바로 풀어버리는 것은 복잡한 함수에 대해서는 적용하기가 어려우며 무엇보다도 Chain Rule의 방법으로 접근하면 각각의 변수에 대한 미분을 sub problem 풀듯이 접근할 수 있다는 장점이 매우 크다.

아래의 3Blue1Brown채널에서 Backpropagation 설명을 보면 가중치에 대한 미분을 Chain Rule을 이용해 접근하는 것을 볼 수 있다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/tIeHLnjs5U8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

딥러닝 프레임워크에서 사용하는 auto differentiation 패키지들은 각 node에서의 편미분값을 저장해 전파(propagate) 시킬 수 있어 복잡한 연산에도 용이하게 적용할 수 있다.

# 4 Conclusion

이번 포스팅에서는 미분하려는 함수가 벡터인 경우에 어떻게 미분하는지를 살펴보았다. 앞서 언급한 것처럼 실제 계산은 대부분은 패키지에서 수행되므로 "함수와 미분하려는 대상이 모두 벡터라면 이런식으로 되는구나" 정도로 이해해도 논문을 읽거나 구현함에 있어 문제가 없다.

# 5 Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.