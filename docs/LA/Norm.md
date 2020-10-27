# Norm

벡터는 방향과 크기를 가지는 값이다. 이번 문서에서는 벡터의 크기는 어떻게 정의되는지에 다룬다.

## Norm

Norm은 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Norm**
>
> 벡터공간 $V$에서 다음의 함수를 생각해보자.
> $$\begin{aligned}\lVert \cdot \rVert: V & \rightarrow \mathbb{R} \\x & \mapsto \lVert x \rVert\end{aligned}$$
> 이 함수를 **norm**이라고 하며 임의의 벡터 $\boldsymbol{x}$에 대해 길이를 $\lVert \boldsymbol{x} \rVert \in \mathbb{R}$로 부여해준다.

> [!NOTE]
> **Definition: $p$-norm**
>
> 실수 $p \geqslant 1$일 때, $p$-norm은 다음과 같이 정의된다.
> $$\lVert \boldsymbol{x} \rVert_{p} = (\lvert x_{1} \rvert^{p} + \lvert x_{2} \rvert^{p} + \cdots + \lvert x_{n} \rvert^{p})^{1/p}$$

참고로, $1$-norm인 경우를 Manhattan norm 또는 Taxicab norm이라고 하며 우리에게 익숙한 $2$-norm은 Euclidean norm이라고 한다. $p=\infty$인 경우는 maximum norm이라고 하며 $\lVert \boldsymbol{x} \rVert_{\infty} = \operatorname{max} \{ \lvert x_1 \rvert, \lvert x_2 \rvert, \ldots, \lvert x_n \rvert \}$와 같다.

## Properties

Norm은 다음의 성질을 만족한다.

* Absolutely homogeneous: $\lVert \lambda \boldsymbol{x} \rVert = \lvert \lambda \rvert \lVert \boldsymbol{x} \rVert$
* Triangle inequality: $\lVert \boldsymbol{x} + \boldsymbol{y} \rVert \leqslant \lVert \boldsymbol{x} \rVert + \lVert \boldsymbol{y} \rVert$
* Positive definite: $lVert \boldsymbol{x} \rVert \geqslant 0$ and $ \lVert \boldsymbol{x} \rVert = 0 \iff \boldsymbol{x} = 0$

## Conclusion

Norm은 벡터공간에서 길이를 정의하는 중요한 역할을 하며, 머신러닝에서 이 개념은 regularization에서도 활용된다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
