# Gradients of Matrices

지금까지 벡터를 벡터로 미분하는 것까지 다루었다. 벡터와 벡터의 미분은 최대 2차원 Jacobian으로 표현이 가능했다. 여기서 다루는 행렬의 미분은 Jacobian이 3차원 이상의 텐서로 표현된다.

## Gradients of Matrices

데이터는 행렬로 표현되고 행렬을 미분해야 하는 상황은 머신러닝에서 흔하게 발생한다. 행렬에 대한 미분은 어떻게 표현되는지를 알아보자. 예를 들어 $m \times n$의 크기를 갖는 $\boldsymbol{A}$를 $p \times q$의 크기를 갖는 $\boldsymbol{B}$로 미분하면, Jacobian $\boldsymbol{J}$의 크기는 $ (m \times n) \times (p \times q) $로 4차원 텐서가 된다. 또한 $ \boldsymbol{J} $의 원소 $J_{ijkl} = \frac{\partial A_{ij}}{\partial B_{kl}}$이다.

다음과 같은 $\boldsymbol{A} \in \mathbb{R}^{4 \times 2}$의 행렬을 $\boldsymbol{x} \in \mathbb{R}^{3}$으로 미분한다고 해보자.

<figure align=center>
<img src="assets/images/VC/Fig_5.7(0).png"/>
<figcaption>Fig 5.7</figcaption>
</figure>

행렬의 미분은 다음의 두 가지 관점으로 바라볼 수 있다. 첫번째 관점은 직관적으로 이해하기 쉬운 형태로 각 원소에 대해 미분한 행렬을 쌓아올리는 방식이다.

<figure align=center>
<img src="assets/images/VC/Fig_5.7(a).png" width=50% height=50%/>
<figcaption>Fig 5.7</figcaption>
</figure>

위 그림을 보면 행렬을 $x_{i}$에 대해서 각각 편미분하고 Fig5.7의 3이 가리키는 방향으로 합치는 것을 볼 수 있다.

다른 방식은 행렬을 위와 같이 바로 미분하지 않고 벡터로 변환한 뒤(이러한 과정을 flatten이라고 한다) 벡터-벡터 미분으로 바꾸는 방식이다. 예를 들어 $m \times n$의 크기를 갖는 행렬은 $mn$의 크기를 갖는 벡터로 reshape할 수 있다. 이러한 방식을 그림으로 나타내면 다음과 같다.

<figure align=center>
<img src="assets/images/VC/Fig_5.7(b).png" width=50% height=50%/>
<figcaption>Fig 5.7</figcaption>
</figure>

$4 \times 2$의 크기를 갖는 행렬 $\boldsymbol{A}$를 크기 $8$의 벡터로 바꾸어 주면 벡터-벡터 미분으로 바뀌게 되고 $8 \times 3$의 Jacobian을 얻을 수 있다. 여기서 첫 번째 차원인 $8$을 다시 원상복구해 $4 \times 2 \times 3$으로 표현하면 같은 결과를 얻을 수 있다.

이해하기에는 첫번째 방식이 더 쉬우나 실제 컴퓨터를 통한 연산을 할 때는 chain rule을 적용하기가 용이한 두 번째 방식이 선호된다.

## Useful Identities for Computing Gradients

미적분학의 미분법칙이 있듯이 vector calculus에도 미분법칙이 있다. 다음은 자주 사용되는 행렬/벡터의 미분공식이다.

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^{\top} &= \left( \frac{\partial \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} \right)^{\top} \\
\frac{\partial}{\partial \boldsymbol{X}} \text{tr}(\boldsymbol{f}(\boldsymbol{X})) &= \text{tr} \left( \frac{\partial \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} \right) \\
\frac{\partial}{\partial \boldsymbol{X}} \text{det}(\boldsymbol{f}(\boldsymbol{X})) &= \text{det}(\boldsymbol{f}(\boldsymbol{X})) \text{tr} \left( \boldsymbol{f}(\boldsymbol{X})^{-1} \frac{\partial \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} \right) \\
\frac{\partial}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^{-1} &= -\boldsymbol{f}(\boldsymbol{X})^{-1} \frac{\partial \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^{-1} \\
\frac{\partial \boldsymbol{a}^{\top} \boldsymbol{X}^{-1} \boldsymbol{b}}{ \partial \boldsymbol{X} } &= -( \boldsymbol{X}^{-1} )^{\top} \boldsymbol{a} \boldsymbol{b}^{\top} ( \boldsymbol{X}^{-1} )^{\top} \\
\frac{\partial \boldsymbol{x}^{\top} \boldsymbol{a}}{\partial \boldsymbol{x}} &= \boldsymbol{a}^{\top} \\
\frac{\boldsymbol{a}^{\top} \boldsymbol{x}}{\partial \boldsymbol{x}} &= \boldsymbol{a}^{\top} \\
\frac{\partial \boldsymbol{a}^{\top} \boldsymbol{X} \boldsymbol{b}}{\partial \boldsymbol{X}} &= \boldsymbol{a} \boldsymbol{b}^{\top} \\
\frac{\partial \boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}}{\partial \boldsymbol{x}} &= \boldsymbol{x}^{\top} (\boldsymbol{B} + \boldsymbol{B}^{\top}) \\
\frac{\partial}{\partial \boldsymbol{x}} (\boldsymbol{x} - \boldsymbol{A} \boldsymbol{s})^{\top} \boldsymbol{W} (\boldsymbol{x} - \boldsymbol{As}) &= -2(\boldsymbol{x} - \boldsymbol{As})^{\top} \boldsymbol{WA} \ \text{for symmetric} \ \boldsymbol{W}
\end{aligned}
$$

## Conclusion

이번 문서에서는 미분의 범위를 행렬의 미분까지 확장하였다. 행렬이라 하더라도 결국 벡터미분에 적용한 원리를 동일하게 적용하여 개념적으로 크게 다르지는 않다. 또한 행렬을 미분할 때 컴퓨터가 다루기 편하도록 reshape을 한 뒤 벡터의 미분으로 바꾸어 계산한다는 점을 알 수 있었다. 행렬의 미분은 꽤나 복잡한 형태를 보이고 미분공식도 직관적으로 이해하기가 어렵다. 하지만 실무적으로는 auto-differentiation library를 사용하기 때문에 이러한 미분을 직접하는 일은 없으므로 편하게 받아들여도 될 것 같다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.