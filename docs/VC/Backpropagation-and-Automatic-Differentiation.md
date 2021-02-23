# Backpropagation and Automatic Differentiation

이번 포스팅에서는 본격적으로 TensorFlow나 PyTorch와 같은 딥러닝 라이브러리에서 신경망을 학습시키는 원리에 대해 다룬다. 예로, TensorFlow에서 `apply_gradient`라는 method를 호출했을 때 어떻게 trainable parameter를 업데이트 하는지, 그리고 이를 효율적으로 가능하게 해주는 automatic differentiation이란 무엇인지가 핵심 주제이다.

## Backpropagation

Machine learning model을 훈련하는 과정에서 gradient descent기반의 optimization을 하는 경우가 많다. Objective function을 model의 파라미터로 미분하여 기존 값에 더하거나(ascent) 빼는(descent)방식을 반복적으로 적용하는 것이다. 이번 포스팅에서 computation graph를 사용해 gradient를 표현하는 방법을 소개하는데, 비교해서 보기 위해 우선 Chain Rule을 사용해 gradient를 explicit하게 표현한 경우를 살펴보자.

예를 들어, $x$를 변수로 하는 목적함수 $f$가 다음과 같이 있다고 해보자.

$$
f(x) = \sqrt{x^{2} + \exp(x^2)} + \cos \left( x^2 + \exp(x^2) \right)
$$

이 때 $x$에 대한 미분은 다음과 같다.

$$
\begin{aligned}
\frac{df}{dx} &= \frac{2x + 2x \exp(x^2)}{2 \sqrt{x^2 + \exp(x^2)}} - \sin(x^2 + \exp(x^2)) (2x + 2x \exp(x^2)) \\
&= 2x \left( \frac{1}{2 \sqrt{x^2} + \exp(x^2)} - \sin (x^2 + \exp(x^2)) \right) (1 + \exp(x^2))
\end{aligned}
$$

보이는 것처럼 이를 실제 계산에 활용하기에는 몇 가지 문제점이 발생하는데, 우선 미분식이 매우 장황해지게 된다. 이렇게 해석적으로 풀어버리면 바로 계산할 수 있다는 장점이 있지만 수 많은 변수들을 연달아 chain rule로 계산하는 관점에서는 좋은 접근이 아니다. 효율적으로 계산하기 위해서는 이전 계산의 결과를 다음 계산에 사용하는 형태가 선호된다. 이름에서 나타나듯 계산을 전파(propagate)시켜 효율적으로 계산하려는 것이 핵심 아이디어이다.

이제 이러한 문제점들을 해결한 backpropagation algorithm을 알아보자. 참고로 이 backpropagation algorithm은 무려 1960년대 초반에 제시된 알고리즘이다.

## Gradients in a Deep Network

Chain rule을 적극적으로 활용하는 분야 중 하나가 딥러닝이다. 여러 레이어를 통과하는 딥러닝 방법은 여러 함수의 합성함수로 바라볼 수 있다.

$$
\begin{aligned}
\boldsymbol{y} &= (f_{K} \circ f_{K-1} \circ \cdots \circ f_{1})(\boldsymbol{x}) \\
&= f_{K} (f_{K-1} ( \cdots (f_{1}(\boldsymbol{x})) \cdots ))
\end{aligned}
$$

그림을 나타내면 다음과 같다.

<figure align=center>
<img src="assets/images/VC/Fig_5.8.png" width=100% height=100%/>
<figcaption>Fig 5.8</figcaption>
</figure>

$i$번째 레이어를 통과한 결과는 $ f_{i}(\boldsymbol{x_{i-1}}) = \sigma( \boldsymbol{A}_{i-1} \boldsymbol{x}_{i-1} + \boldsymbol{b}_{i-1}) $로 표현할 수 있다. 여기서 $\sigma$는 activation function이다. 신경망의 weight을 업데이트하기 위해서는 loss function $L$의 $\boldsymbol{A}_{j}, \boldsymbol{b}_{j}, j = 1, \ldots, K$에 대한 gradient를 모두 알아야 한다.

각 레이어의 계산을 일반적으로 표현하기 위해 위의 식을 아래와 같이 바꾸어보자.

$$
\begin{aligned}
\boldsymbol{f}_{0} &:= \boldsymbol{x} \\
\boldsymbol{f}_{i} &:= \sigma_{i} (\boldsymbol{A}_{i-1} \boldsymbol{f}_{i-1} + \boldsymbol{b}_{i-1}), i = 1, \ldots, K
\end{aligned}
$$

만약, loss function이 squared loss라면,

$$
L(\boldsymbol{\theta}) = \lVert \boldsymbol{y} - \boldsymbol{f}_{K}(\boldsymbol{\theta}, \boldsymbol{x}) \rVert^{2}
$$

을 최소화하면 된다. 그리고 이 때 $ \boldsymbol{\theta} = { \boldsymbol{A}_{0}, \boldsymbol{b}_{0}, \ldots, \boldsymbol{A}_{K-1}, \boldsymbol{b}_{K-1} \\} $이다.

이제 $\boldsymbol{\theta}$에 대해 gradient를 계산해주면 된다. $\boldsymbol{\theta}_{j} = \\{\boldsymbol{A}_{j}, \boldsymbol{b}_{j}\\}, j=0,\ldots,K-1$라고 정의하면 각 $\boldsymbol{\theta}$에 대한 gradient는 다음과 같이 계산된다.

$$
\begin{aligned}
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-1}} &= \frac{\partial L}{\partial \boldsymbol{f}_{K}} \color{blue}{\frac{\partial \boldsymbol{f}_{K}}{\partial \boldsymbol{\theta}_{K-1}}} \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-2}} &= \frac{\partial L}{\partial \boldsymbol{f}_{K}} \color{orange}{\frac{\partial \boldsymbol{f}_{K}}{\partial \boldsymbol{f}_{K-1}}} \color{blue}{\frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{\theta}_{K-2}}} \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-3}} &= \frac{\partial L}{\partial \boldsymbol{f}_{K}} \color{orange}{\frac{\partial \boldsymbol{f}_{K}}{\partial \boldsymbol{f}_{K-1}}} \color{orange}{\frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{f}_{K-2}}} \color{blue}{\frac{\partial \boldsymbol{f}_{K-2}}{\partial \boldsymbol{\theta}_{K-3}}} \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{i}} &= \frac{\partial L}{\partial \boldsymbol{f}_{K}} \color{orange}{\frac{\partial \boldsymbol{f}_{K}}{\partial \boldsymbol{f}_{K-1}}} \color{orange}{\cdots} \color{orange}{\frac{\partial \boldsymbol{f}_{i+2}}{\partial \boldsymbol{f}_{i+1}}} \color{blue}{\frac{\partial \boldsymbol{f}_{i+1}}{\partial \boldsymbol{\theta}_{i}}}
\end{aligned}
$$

여기서 주황색으로 표시된 항들을 살펴보면 특정 레이어를 이전 레이어에 대한 편미분으로 나타내고 있음을 볼 수 있다. 그리고 가장 오른쪽의 파란색으로 표시된 항은 레이어에 대해서 입력 파라미터 $\boldsymbol{\theta}$로의 편미분을 나타낸다.

위 식에서 볼 수 있듯, $\partial L/\partial \boldsymbol{\theta}_{i}$를 계산할 때 대부분의 식은 이전 레이어의 식을 활용하면 된다. 여기서 back propagation은 가장 마지막 레이어에서 input레이어로 전파된다.

<figure align=center>
<img src="assets/images/VC/Fig_5.9.png" width=100% height=100%/>
<figcaption>Fig 5.9</figcaption>
</figure>

따라서 $ \frac{\partial L}{\partial \boldsymbol{\theta}_{i+1}} $을 이미 계산했다면 이전 레이어($ \frac{\partial L}{\partial \boldsymbol{\theta}_{i}} $)에 대한 gradient를 계산할때는 위 식의 오른쪽 두 개의 항만 추가로 계산하면 된다. 나머지 항들은 이미 계산된 값을 caching해 바로 사용할 수 있다.

Backpropagation의 미덕은 gradient를 계산함에 있어서 마지막 레이어에서 계산한 gradient값을 재사용하면서 앞의 레이어까지 gradient값을 효율적으로 계산한다는데 있다.

## Automatic Differentitation

결과적으로 backpropagation은 수치해석학에서 automatic differentitation의 특수한 경우이다. Automatic differentiation은 symbolic(특정 변수에 대해 정확하게 풀어내는 것으로 숫자를 이용하지 않고 $x, y, z$와 같은 변수에 대해 풀어내는 경우가 이에 해당한다)한 접근이 아닌, 수치적으로 풀어낼 수 있는 방법으로 매개변수(intermediate variables)와 chain rule을 사용해 특정 함수의 gradient를 구하는 것을 말한다.

예를 들어 $x, y$가 매개변수 $a, b$로 표현되어있다고 해보자. 

<figure align=center>
<img src="assets/images/VC/Fig_5.10.png" width=50% height=50%/>
<figcaption>Fig 5.10</figcaption>
</figure>

이 때, $\frac{dy}{dx}$는 chain rule을 사용해 다음과 같이 표현할 수 있다.
$$
\frac{dy}{dx} = \frac{dy}{db} \frac{db}{da} \frac{da}{dx}
$$
계산방향에 따라 forward mode와 reverse mode를 정의할 수 있고 associativity가 성립하므로 forward mode는 다음과 같이 나타낼 수 있다.
$$
\frac{dy}{dx} = \frac{dy}{db} \left( \frac{db}{da} \frac{da}{dx} \right)
$$
Reverse mode는 다음과 같다.
$$
\frac{dy}{dx} = \left( \frac{dy}{db} \frac{db}{da} \right) \frac{da}{dx}
$$

Gradient의 계산은 reverse mode를 따라가면 된다. 그렇다고 해서 forward mode로 계산을 할 수 없는 것은 아니다. Forward mode에서도 계산값을 재사용하며 gradient를 계산할 수 있다. 다만 대부분의 경우 입력레이어의 차원은 출력레이어의 차원보다 훨씬 크다. 가능한한 이미 계산한 결과를 많이 재사용해야 효율적임을 감안할 때, reverse mode가 선호되는 것이다.

### Example

다음은 computation graph를 활용해 실제 automatic differentiation이 어떻게 일어나는지를 설명하는 예제다. 예제로 제시된 함수는 이번 포스팅의 제일 처음 소개된 함수 $f$와 동일한 함수이며 교재 예제 5.14에 해당한다.

$$
f(x) = \sqrt{x^{2} + \exp(x^2)} + \cos \left( x^2 + \exp(x^2) \right)
$$

이를 컴퓨터에서 계산하기 위해서는 계산의 최소 단위(함수)를 모두 매개변수로 바꾸어 표현해주어야 한다. 매개변수들을 표현하면 다음과 같다.

$$
\begin{aligned}
a &= x^2 \\
b &= \exp(a) \\
c &= a + b \\
d &= \sqrt{c} \\
e &= \cos(c) \\
f &= d + e
\end{aligned}
$$

식 기준으로 가장 바깥에서 부터 안쪽으로 점점 작은 단위의 매개변수를 선언한다고 보면된다. 위의 과정은 computation graph로 표현할 때 다음과 같다.

<figure align=center>
<img src="assets/images/VC/Fig_5.11.png" width=100% height=100%/>
<figcaption>Fig 5.11</figcaption>
</figure>

구현측면에 있어서 이를 더 깊게 이해해야 한다.TensorFlow나 PyTorch에서 계산을 빠르게 하기 위해서는 이 Graph를 이용하게 된다. PyTorch는 dynamic graph를 사용하지만 TensorFlow는 static graph를 사용하므로 TensorFlow에서는 계산에서 사용할 graph를 컴파일하고 이 그래프를 반복해 이용하면서 계산을 빠르게 하게 된다. TensorFlow 코드를 작성할 때 warning으로 그래프를 너무 많이 형성한다고 나올 때가 있는데, 이는 재사용하여야 하는 graph가 구현상의 실수로 매번 새로운 graph를 만들게 되었을 때 주로 발생한다. 한 번 컴파일하고 이를 데이터 값만 다르게 재사용해야 하는데 매번 새로운 그래프를 만들도록 프로그래밍 되는게 자주 발생하는 실수 중 하나이다.

이제 위의 computation graph를 보면서 인접한 두 node의 편미분 식을 정리하면 다음과 같다.

$$
\begin{aligned}
\frac{\partial a}{\partial x} &= 2x \\
\frac{\partial b}{\partial a} &= \exp(a) \\
\frac{\partial c}{\partial a} &= 1 = \frac{\partial c}{\partial b} \\
\frac{\partial d}{\partial c} &= \frac{1}{2 \sqrt{c}} \\
\frac{\partial e}{\partial c} &= -\sin(c) \\
\frac{\partial f}{\partial d} &= 1 = \frac{\partial f}{\partial e}
\end{aligned}
$$

$\frac{\partial f}{\partial x}$는 computation graph를 사용해 다음과 같이 분해할 수 있다.

$$
\begin{aligned}
\frac{\partial f}{\partial c} &= \frac{\partial f}{\partial d} \frac{\partial d}{\partial c} + \frac{\partial f}{\partial e} \frac{\partial e}{\partial c} \\
\frac{\partial f}{\partial b} &= \frac{\partial f}{\partial c} \frac{\partial c}{\partial b} \\
\frac{\partial f}{\partial a} &= \frac{\partial f}{\partial b} \frac{\partial b}{\partial a} + \frac{\partial f}{\partial c} \frac{\partial c}{\partial a} \\
\frac{\partial f}{\partial x} &= \frac{\partial f}{\partial a} \frac{\partial a}{\partial x}
\end{aligned}
$$

이제 앞에서 계산한 값들을 대입하기만 하면 목표로 했던 $\frac{\partial f}{\partial x}$를 계산할 수 있다. 보기에는 explicit하게 계산하는 것이 더 간단해 보이지만 컴퓨터로 계산할 때는 chain rule을 사용한 computation graph 방식이 훨씬 효율적이다. 그리고 computation graph를 살펴보면 대부분의 계산 결과가 사용되는 미덕을 볼 수 있다.

## Formalization of Automatic Differentitation

이제 위의 결과를 바탕으로 일반화를 해보자. 입력변수가 $ x_{1}, \ldots, x_{d} $이고, $x_{d+1}, \ldots, x_{D-1}$이 매개변수, 그리고 $x_{D}$가 출력변수라고 해보자.

$g_{i} (\cdot)$이 elementary functions, $x_{\text{Pa}(x_{i})}$가 $x_{i}$의 parent nodes라고 할 때, computation graph는 다음과 같이 표현할 수 있다.

$$
\text{For} \ i=d+1, \ldots, D: \quad x_{i} = g_{i}(x_{\text{Pa} (x_i)})
$$

출력변수가 $x_{D}$라고 하였으므로 다음이 성립한다.

$$
\frac{\partial f}{\partial x_{D}} = 1
$$

다른 변수 $x_{i}$에 대해서 chain rule을 적용하면 다음과 같이 표현할 수 있다.

$$
\frac{\partial f}{\partial x_{i}} = \sum_{x_{j}:x_{i} \in \text{Pa}(x_{j})} \frac{\partial f}{\partial x_{j}} \frac{\partial x_{j}}{\partial x_{i}} = \sum_{x_{j}:x_{i} \in \text{Pa}(x_{j})} \frac{\partial f}{\partial x_{j}} \frac{\partial g_{j}}{\partial x_{i}}
$$

Computation graph의 강점은 어떠한 함수이던 computation graph 형태로 표현만 할 수 있고 elementary function이 미분가능한 형태라면, gradient를 구할 수 있다는 것이다.

## Conclusion

이번 장에서는 backpropagation과 automatic differentiation에 대해 다루었다. 이 개념들은 딥러닝이 어떻게 학습하는 지를 설명해주는 개념들로 직접 계산할 일은 드물겠지만 딥러닝의 학습원리에 관한 내용이므로 반드시 알아둘 필요가 있다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.