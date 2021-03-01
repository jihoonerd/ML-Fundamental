# Higher-Order Derivatives

지금까지 다루었던 gradient는 최대 first-order derivative였다. 하지만 Newton's method와 같은 최적화 방법을 도입하기 위해서는 때때로 second-order derivative이상의 미분을 해야할 때가 있다. 하지만 미분의 원리 자체는 동일하므로 이번 문서에서는 higher-order derivative에 대해 간략히 알아보도록 한다.

## Notation of Higher-Order Derivatives

우선 notation부터 정리하도록 하자. 임의의 함수 두 변수 $x, y$를 갖는 $f: \mathbb{R}^{2} \rightarrow \mathbb{R}$가 있다고 할 때, higher-order partial derivatives는 다음과 같이 표현된다.

* $ \frac{\partial^{2} f}{\partial x^{2}} $: 함수 $f$의 $x$에 대한 2차 편미분
* $ \frac{\partial^{n} f}{\partial x^{n}} $: 함수 $f$의 $x$에 대한 n차 편미분

편미분이 꼭 같은 변수에 대해 될 필요는 없다. $x, y$ 각각에 대해서 편미분 되면 다음과 같이 표기한다.

* $ \frac{\partial^{2} f}{\partial y \partial x} = \frac{\partial}{\partial y} \left(\frac{\partial f}{\partial x}\right) $: 함수 $f$의 $x$에 대한 편미분의 $y$에 대한 편미분
* $ \frac{\partial^{2} f}{\partial x \partial y} = \frac{\partial}{\partial x} \left(\frac{\partial f}{\partial y}\right) $: 함수 $f$의 $y$에 대한 편미분의 $x$에 대한 편미분

## Hessian

Hessian은 모든 second-order partial derivatives의 집합을 의미하며 편미분의 순서가 상관없다면 Hessian matrix로서 다음과 같이 표현할 수 있다.

$$
\boldsymbol{H} = 
\begin{bmatrix}
\frac{\partial^{2} f}{\partial x^{2}} & \frac{\partial^{2} f}{\partial x \partial y} \\
\frac{\partial^{2} f}{\partial x \partial y} & \frac{\partial^{2} f}{\partial y^{2}}
\end{bmatrix}
$$

Hessian matrix는 symmetric하며 $ \nabla_{x, y}^{2} f(x, y) $로 표기할 수도 있다. 일반적으로 $\boldsymbol{x} \in \mathbb{R}^{n}, f: \mathbb{R}^{n} \rightarrow \mathbb{R}$일 때, Hessian은 $n \times n$의 square matrix이다. 기하학적으로 Hessian은 $(x,y)$근방에서의 곡률을 의미한다.

만약 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$으로 vector-field에서의 함수라면, Hessian은 $ (m \times n \times n) $ 크기의 tensor로 표현된다.

## Conclusion

이번 문서에서는 second-order이상의 편미분을 어떻게 표기하는지와, 2차 편미분을 나타내는 Hessian에 대해 알아보았다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.