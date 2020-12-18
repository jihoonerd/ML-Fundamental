# Cholesky Decomposition

모든 연산이 그렇지는 않지만 스칼라에서 사용하는 많은 연산들은 행렬의 연산에 대응한다. 역원, 항등원은 물론 제곱과 같은 연산도 행렬에 대응하는 개념이 있다. 그렇다면 제곱근은 어떨까? 한 행렬을 같은 두 행렬의 곱으로 분해할 수 있을까? 이것이 Cholesky decomposition이 다루는 주제이다.

## Cholesky Decomposition

직관적으로 생각하면 행렬의 제곱근이 Cholesky decomposition이다. 모든 행렬에 적용할 수 있을까? 일반 숫자에서 제곱근도 실수영역내에서는 제곱근이 존재할 조건을 가지듯, 행렬도 마찬가지이다. 우리는 주로 실수공간에 관심이 있으므로 실수공간에 한정할 때 Cholesky decomposition이 가능한 조건이 정해진다.

> [!NOTE]
> **Theorem: Cholesky Decomposition**
>
> **symmetric, positive definite matrix** $\boldsymbol{A}$는 $\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}^\top$로 분해할 수 있으며 이 때$\boldsymbol{L}$은 양수인 대각성분을 갖는 lower triangular matrix이다.
> $$\begin{bmatrix}a_{11} & \cdots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{n1} & \cdots & a_{nn} \end{bmatrix} = \begin{bmatrix}l_{11} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ l_{n1} & \cdots & l_{nn}\end{bmatrix} \begin{bmatrix}l_{11} & \cdots & l_{n1} \\ \vdots & \ddots & \vdots \\ 0 & \cdots & l_{nn}\end{bmatrix}$$
> $\boldsymbol{L}$을 $\boldsymbol{A}$의 **Cholesky factor**라고 하며 $\boldsymbol{L}$은 $\boldsymbol{A}$에 대해 유일하게 존재한다.

Cholesky decoposition은 multivariate Gaussian variable의 covariance matrix로 부터 새로운 sample을 생성하는데 사용될 수 있고 variational autoencoder의 gradient계산에 사용할 수 있다고 한다.

계산 측면에 있어서도 Cholesky decomposition은 매우 유용하다. 예를 들어 determinant를 구한다고 할 때 Cholesky decomposition의 정의에 의해

$$
\text{det}(\boldsymbol{A}) = \text{det}(\boldsymbol{L})\text{det}(\boldsymbol{L}^\top) = \text{det}(\boldsymbol{L})^2
$$

이 성립한다. 이 때 $\boldsymbol{L}$은 triangular matrix이므로 determinant를 구할 때 대각성분만 곱하면 바로 얻을 수 있어 컴퓨터에서 determinant를 계산할 때 이를 활용하기도 한다. 실제로, Cholesky decomposition은 LU decomposition에 비해 선형방정식을 푸는데 약 2배 빠르다고 알려져있다.

## Conclusion

이번 문서에서는 Cholesky decomposition에 대해서 간단하게 알아보았다. 교재에서 직접언급되지는 않았지만 LU Decomposition에 비해 더 간결하게 표현되는 만큼 symmetric하고 positive definite해야한다는 제한이 있다. LU decomposition이나 Cholesky decompoistion으로 분해가 되었다면 linear equation과 determinant, trace를 구하는 것은 매우 간단해진다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.