# Angle and Orthogonality

앞서 내적을 이용해 벡터내적공간에서 길이와 거리에 대한 개념을 정의했었다. 이번 문서에서는 각도에 대해 정의를 하고 특히 직각을 이루는 직교성에 대해 다룬다. 내적공간에서 measure들이 확장되어가는 것에 유의해 각도의 정의를 살펴보자.

## Angle between Two Vectors

앞에서 정의한 Cauchy-Schwartz 부등식을 다시 살펴보자.

> [!NOTE]
> **Definition: Cauchy-Schwarz Inequality**
> 
> 내적공간 $(V, \langle \cdot, \cdot \rangle)$이 유도하는 norm $\lVert \cdot \rVert$는 다음의 **Cauchy-Schwartz inequality** 를 만족한다.
> $$\lvert \langle \boldsymbol{x}, \boldsymbol{y} \rangle \rvert \leqslant \lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert$$

Cauchy-Schwartz 부등식에서 $\boldsymbol{x} \neq 0, \boldsymbol{y} \neq 0$이면 다음과 같이 표현할 수 있다.

$$
-1 \leq \frac{\langle \boldsymbol{x}, \boldsymbol{y} \rangle}{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert} \leq 1
$$

두 벡터 $\boldsymbol{x}, \boldsymbol{y}$가 이루는 각도를 $\omega$라고 하면, $\omega \in [0, \pi]$는 $\cos$함수에 대해 일대일대응이다. 따라서 두 벡터 $\boldsymbol{x}, \boldsymbol{y}$가 이루는 각도 $\omega$는 다음과 같이 정의한다.

> [!NOTE]
> **Definition: Angle**
>
> 두 벡터 $\boldsymbol{x}, \boldsymbol{y}$에 대해서 다음을 두 벡터가 이루는 **각도(angle)** 로 정의한다.
> $$\cos{\omega} = \frac{\langle \boldsymbol{x}, \boldsymbol{y} \rangle}{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert}$$

내적으로 정의되어 있음에 유의한다. Dot product를 내적으로 사용할 때 우리가 익숙하게 사용하는 Euclidean 공간에서의 각도가 된다.

## Orthogonality

선형대수학에서 직교성(Orthogonality)는 중요하게 다루어지는 주제이다. 당장 머리속에 2차원 혹은 3차원 공간을 떠올린다면 직교하는 축으로 정의된 공간을 떠올리게 된다. 그런데 굳이 축들이 직교할 필요가 있을까? 직교하지 않는 기저들로도 충분히 벡터공간을 표현할 수 있다. 뒤에 다루는 내용이지만 직교하지 않는 경우 Gram-Schmidt Process까지 써가며 굳이 직교벡터를 찾는 이유는 무엇일까? 직교성의 정의와 직교벡터가 갖는 장점에 대해 알아보자.

직교는 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Orthogonality**
>
> 두 벡터 $\boldsymbol{x}, \boldsymbol{y}$가 **직교성(Orthogonal)** 을 갖는다는 것은 $\langle \boldsymbol{x}, \boldsymbol{y} \rangle = 0$인 것과 필요충분조건이다. 그리고 두 벡터가 직교할 때 $\boldsymbol{x} \perp \boldsymbol{y}$로 표현한다. 특히, $\lVert \boldsymbol{x} \rVert = 1 = \lVert \boldsymbol{y} \rVert$까지 만족하면 $\boldsymbol{x}, \boldsymbol{y}$는 **orthonormal** 이라고 한다.

정의에 의해 $\boldsymbol{0}$는 모든 벡터와 직교한다. 하지만 직교성의 정의는 정의에서 확인되는 것처럼 내적에 대해 정의가 되는 것이지 각도와 마찬가지로 dot product에 대해 정의가 되는 것이 아니다. 

예로, $\boldsymbol{x} = [1, 1]^\top, \boldsymbol{y} = [-1, 1]^\top \in \mathbb{R}^2$인 두 벡터는 2차원 평면에서 생각할 때 직교한다. 하지만 이 두 벡터의 직교란 어디까지나 dot product를 내적으로 사용할 때에 한해서이다. 내적이 다음과 같이 정의되었다면,

$$
\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^\top
\begin{bmatrix}
2 & 0 \\
0 & 1
\end{bmatrix}
\boldsymbol{y}
$$

각도는 정의에 의해 다음으로 계싼할 수 있다.

$$
\cos \omega = \frac{\langle \boldsymbol{x}, \boldsymbol{y} \rangle}{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert} = -\frac{1}{3}
$$

이 등식을 만족하는 $\omega$는 약 $109.5^\circ$이다. 따라서 내적이 0이 아니므로 내적이 위와 같이 정의되었다면 두 벡터는 직교하지 않는다.

이를 통해 한 내적에 대해서 직교한다는 것이 다른 내적에 대해서도 직교함을 의미하지는 않는다는 것도 확인할 수 있다.

## Orthogonal Matrix

> [!NOTE]
> **Definition: Orthogonal Matrix**
>
> 어떤 정사각행렬 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$의 열벡터들이 orthonormal하면 **Orthogonal Matrix**라고 하며 다음이 성립한다.
> $$\boldsymbol{A}\boldsymbol{A}^\top = \boldsymbol{I} = \boldsymbol{A}^\top \boldsymbol{A}$$
> 그리고 역행렬의 정의에 의해 $\boldsymbol{A}^{-1} = \boldsymbol{A}^\top$임을 알 수 있다. 따라서 역행렬은 단순하게 Transpose만 해도 얻을 수 있다.

이름이 orthonormal matrix면 더 직관적이겠다는 생각이들지만 orthogonal matrix성질상 orthonormal의 성질을 함의하므로 그러려니 생각하자. Quora에도 [관련 질문](https://www.quora.com/Why-don-t-we-call-orthogonal-matrix-just-orthonormal-matrix-if-its-columns-rows-are-orthonormal#:~:text=Because%20in%20LA%2C%20the%20word,an%20orthonormal%20set%20of%20vectors.&text=Why%20does%20a%20matrix%20have,to%20get%20an%20inverse%20matrix%3F)이 있다.

내적을 dot product로 정의할 때, Orthogonal matrix에 의한 변환은 벡터의 길이를 보존하고, 어떤 두 벡터에 대해서 각각 변환을 하여도 두 벡터가 이루는 각도가 보존된다는 특별한 성질이 있다. 하나씩 살펴보자.

### Length Invariance

임의의 벡터 $\boldsymbol{x}$에 대해서 orthogonal matrix에 의한 변환은 길이가 보존된다는 성질이 있다.

$$\begin{aligned} \lVert \boldsymbol{A} \boldsymbol{x} \rVert^{2} &= (\boldsymbol{A} \boldsymbol{x})^{\top}(\boldsymbol{A} \boldsymbol{x}) \\ &=\boldsymbol{x}^{\top} \boldsymbol{A}^{\top} \boldsymbol{A} \boldsymbol{x} \\ &= \boldsymbol{x}^{\top} \boldsymbol{I} \boldsymbol{x} \\ &= \boldsymbol{x}^{\top} \boldsymbol{x} \\ &=\lVert \boldsymbol{x}\rVert^{2} \end{aligned}$$

### Angle Invariance

임의의 벡터 $\boldsymbol{x}, \boldsymbol{y}$가 이루는 각도는 각각의 벡터를 orthogonal matrix $\boldsymbol{A}$로 변환한하여도 보존된다.

$$\begin{aligned} \cos \omega &= \frac{ (\boldsymbol{Ax}^{\top}) (\boldsymbol{Ay}) }{\lVert \boldsymbol{Ax} \rVert \lVert \boldsymbol{Ay} \rVert} \\ &= \frac{\boldsymbol{x}^{\top} \boldsymbol{A}^{\top} \boldsymbol{A} \boldsymbol{y}}{\boldsymbol{x}^{\top} \boldsymbol{A}^{\top} \boldsymbol{Ax} \boldsymbol{y}^{\top} \boldsymbol{A}^{\top} \boldsymbol{A} \boldsymbol{y}} \\ &= \frac{\boldsymbol{x}^{\top} \boldsymbol{y}}{\lVert \boldsymbol{x} \rVert \lVert \boldsymbol{y} \rVert} \end{aligned}$$

길이와 각도의 보존은 위와 같이 정의식을 이용해서 간단하게 보일 수 있다. 기하적으로는 어떤 의미가 있을까? 공간에서 어떠한 변환을 했을 때 길이와 각도가 보존되는 변환은 무엇이 있을까? 결과적으로 orthogonal matrix가 정의하는 변환은 공간에서 회전에 해당하는 변환이다. 직관적으로 생각해보아도 이러한 성질은 회전변환에서 만족된다는 것을 쉽게 예상해볼 수 있다. 이 내용은 이후 회전을 다룰 때 자세히 다루게 된다.

## Conclusion

이전 문서에서는 길이와 거리개념을 내적공간에서 정의하였고 이 문서에서는 각도와 직교성, orthogonal matrix의 정의와 성질을 알아보았다. 

지금까지의 흐름을 보면, 벡터공간을 정의해 벡터공간에서 성립하는 연산들을 알아보았고 내적이 정의되는 내적공간을 만들고 이를 바탕으로 길이와 각도와 같은 개념들이 정의되고 있다는 것을 알 수 있을 것이다. 선형대수학에서 다루는 공간들이 어떤 공간인지 이해하여야 이후 다루는 내용들을 더 명확히 이해할 수 있으므로 익숙한 개념이지만 선형대수학의 관점에서 이해해보도록 하자.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
