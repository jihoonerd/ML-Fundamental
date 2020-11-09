# Orthogonal Projection

머신러닝에서 다루는 데이터는 대부분 3차원 이상의 고차원데이터로, 사람이 인지할 수 있는 3차원이하의 공간으로는 데이터를 표현하기 어렵다는 문제가 있다. 하지만 데이터의 모든 정보가 중요한 경우는 많지 않다. 따라서, 필요없는 변수는 줄이고 가능한 한 중요한 정보만을 보존하는 변수를 활용하면 고차원의 데이터를 시각화가 가능한 영역까지 축소할 수 있다. 그렇다면 compression loss를 최소화하면서 중요한 정보를 보존하려면 어떻게 해야할까? 이러한 논의를 하기 위해 기초가 되는 projection에 대해 알아보자.


## Projection

Projection의 정의는 다음과 같다.

> [!NOTE]
> **Definition: Projecdtion**
>
> Vector space $V$에 대해 $U$가 $V$의 Subspace라고 할 때 ($U \subseteq V$), Linear mapping $\pi: V \rightarrow U$가 $\pi^2 = \pi \circ \pi = \pi$를 만족하면 **Projection**이라고 한다.

직관적으로 생각해보면 projection은 벡터공간에서 특정 벡터공간으로의 함수이므로 이미 특정공간으로 projection하였다면 이에 대해 projection을 다시 하더라도 원래 projection과 결과가 같다.

Linear mapping은 transformation matrix로 표현할 수 있으므로 projection은 transformation matrix의 특별한 경우라고 할 수 있다. 선형변환을 표현하는 transformation matrix로 이 성질은 다음과 같이 쓸 수 있다.

$$
\boldsymbol{P}_{\pi}^2 = \boldsymbol{P}_{\pi}
$$

이제부터 벡터의 orthogonal projection을 inner product space ($\mathbb{R}^n, \langle \cdot, \cdot \rangle$)에서 벡터부분공간의 함수로 다루게 된다. 거창하게 표현되었지만 쉽게 표현하면 $n$차원 데이터와 내적이 정의된 공간을 벡터부분공간으로 보내는 함수에 관한 내용이다. 아래의 projection 내용을 다룰 때 별도의 언급이 없는 한 내적은 dot product $\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^\intercal \boldsymbol{y}$라고 가정한다.

### Projection onto One-Dimensional Subspaces (Lines)

원점을 지나는 기저벡터 $\boldsymbol{b} \in \mathbb{R}^n$가 있을 때 $\operatorname{span}(\boldsymbol{b})$인 직선은 1차원 부분공간 $U \subseteq \mathbb{R}^n$로 나타낼 수 있다. $n$차원 공간의 임의의 벡터 $\boldsymbol{x} \in \mathbb{R}^n$를 직선으로 표현된 부분공간 $U$로 projection하려면 $U$공간에 있되, 가장 $\boldsymbol{x}$와 가까운 $\pi_{U}(\boldsymbol{x}) \in U$를 찾으면 된다. 이 때 $\pi_{U}(\boldsymbol{x})$의 성질은 다음과 같다.

* $\pi_{U}(\boldsymbol{x})$는 $\boldsymbol{x}$와 가장 가까우며, 여기서 "가깝다"는 것은 $\lVert \boldsymbol{x} - \pi_{U}(\boldsymbol{x}) \rVert$가 가장 작을 때를 의미한다. 다시 말해, Euclidean 공간에서 $\boldsymbol{x}$을 $\operatorname{span}(\boldsymbol{b})$에 수직선을 내렸을 때의 거리이다. 이는 다음과 동치이다.

  $$
  \langle \pi_{U}(\boldsymbol{x}) - \boldsymbol{x}, \boldsymbol{b} \rangle = 0
  $$

* $\boldsymbol{x}$의 $U$위로의 projection $\pi_{U}(\boldsymbol{x})$은 $U$공간의 원소이며 따라서 기저벡터 $\boldsymbol{b}$의 스칼라곱으로 표현할 수 있다. 
  
  $$\pi_{U}(\boldsymbol{x}) = \lambda \boldsymbol{b}, \ \text{for some } \lambda \in \mathbb{R}$$

#### Calculation

다음의 과정을 통해 $\boldsymbol{x} \in \mathbb{R}^n$를 $U$로 projection하는 $\lambda, \pi_{U}(\boldsymbol{x}) \in U, \boldsymbol{P}_{\pi}$를 구할 수 있다.

1. Finding the coordinate $\lambda$

  Orthogonality 정의에 의해 다음이 성립한다.

  $$\langle \boldsymbol{x} - \pi_{U}(\boldsymbol{x}), \boldsymbol{b} \rangle = 0 \overset{\pi_{U}(\boldsymbol{x}) = \lambda \boldsymbol{b}}{\iff} \langle \boldsymbol{x} - \lambda \boldsymbol{b}, \boldsymbol{b} \rangle = 0$$

  내적의 선형성을 이용하면 다음과 같이 쓸 수 있다.
  
  $$\langle \boldsymbol{x}, \boldsymbol{b} \rangle - \lambda \langle \boldsymbol{b}, \boldsymbol{b} \rangle = 0 \iff \lambda = \frac{\langle \boldsymbol{x}, \boldsymbol{b} \rangle}{\langle \boldsymbol{b}, \boldsymbol{b} \rangle} = \frac{\langle \boldsymbol{b}, \boldsymbol{x} \rangle}{\lVert \boldsymbol{b} \rVert^2}$$

  만약 $\lVert \boldsymbol{b} \rVert = 1$이면 $\lambda$는 단순하게 $\boldsymbol{b}^T \boldsymbol{x}$을 계산함으로써 얻을 수 있다.

2. Finding the projection point $\pi_{U}(\boldsymbol{x}) \in U$

  $\pi_{U}(\boldsymbol{x}) = \lambda \boldsymbol{b}$이므로 

  $$\pi_{U}(\boldsymbol{x}) = \lambda \boldsymbol{b} = \frac{\langle \boldsymbol{b}, \boldsymbol{x} \rangle}{\lVert \boldsymbol{b} \rVert^2} \boldsymbol{b} = \frac{\boldsymbol{b}^\intercal \boldsymbol{x}}{\lVert \boldsymbol{b} \rVert^2} \boldsymbol{b}$$

  Projection의 길이는 다음과 같다.

  $$\lVert \pi_{U}(\boldsymbol{x}) \rVert = \lVert \lambda \boldsymbol{b} \rVert = \lvert \lambda \rvert \lVert \boldsymbol{b} \rVert$$

  나아가, dot product에 대해서는 다음의 관계까지도 성립한다.

  $$\lVert \pi_{U}(\boldsymbol{x}) \rVert = \lvert \cos \omega \rvert \lVert \boldsymbol{x} \rVert$$

3. Finding the projection matrix $\boldsymbol{P}_{\pi}$

  Projection matrix $\boldsymbol{P}_{\pi}$가존재할 때 projection은 다음과 같이 표현된다.

  $$\pi_{U}(\boldsymbol{x}) = \boldsymbol{P}_{\pi}\boldsymbol{x}$$

  위의 유도를 이용하면,

  $$\pi_{U}(\boldsymbol{x}) = \lambda \boldsymbol{b} = \boldsymbol{b}\lambda = \boldsymbol{b} \frac{\boldsymbol{b}^\intercal \boldsymbol{x}}{\lVert \boldsymbol{b} \rVert^2} = \frac{\boldsymbol{b}\boldsymbol{b}^\intercal}{\lVert \boldsymbol{b} \rVert^2} \boldsymbol{x}$$

  로부터

  $$\boldsymbol{P}_{\pi} = \frac{\boldsymbol{b}\boldsymbol{b}^\intercal}{\lVert \boldsymbol{b} \rVert^2}$$
  
  임을 알 수 있다. $\boldsymbol{P}_{\pi}$는 랭크 1의 symmetric matrix이며 $\lVert \boldsymbol{b} \rVert^{2} = \langle \boldsymbol{b}, \boldsymbol{b} \rangle$임에 유의하자.


이 과정을 통해 얻은 projection matrix $\boldsymbol{P}_{\pi}$는 공간 내 임의의 벡터 $\boldsymbol{x} \in \mathbb{R}^{n}$을 원점을 지나는 벡터 $\boldsymbol{b}$의 직선으로 projection하는 변환행렬이 된다.

#### Example

원점을 지나고 벡터 $\boldsymbol{b} = \begin{bmatrix}1 & 2 & 2 \end{bmatrix}^{\intercal}$에 의해 span되는 직선으로 projection시키는 projection matrix를 찾아보자. $\boldsymbol{b}$는 1차원 부분공간의 기저이다.

Projection matrix를 구하는 공식을 위에서 유도하였으므로 공식에 대입만 하면 된다. 대입하면 아래와 같은 행렬을 얻을 수 있다.

$$
\boldsymbol{P}_{\pi}=\frac{\boldsymbol{b} \boldsymbol{b}^{\intercal}}{\boldsymbol{b}^{\intercal} \boldsymbol{b}}=\frac{1}{9}\left[\begin{array}{l}
1 \\
2 \\
2
\end{array}\right]\left[\begin{array}{lll}
1 & 2 & 2
\end{array}\right]=\frac{1}{9}\left[\begin{array}{lll}
1 & 2 & 2 \\
2 & 4 & 4 \\
2 & 4 & 4
\end{array}\right]
$$

실제로 $\begin{bmatrix}1 & 1 & 1\end{bmatrix}^{\intercal}$에 projection matrix를 적용해 $\boldsymbol{b}$ 위로 projection 되었는지 확인해보자. 

$$
\pi_{U}(\boldsymbol{x})=\boldsymbol{P}_{\pi} \boldsymbol{x}=\frac{1}{9}\left[\begin{array}{lll}
1 & 2 & 2 \\
2 & 4 & 4 \\
2 & 4 & 4
\end{array}\right]\left[\begin{array}{l}
1 \\
1 \\
1
\end{array}\right]=\frac{1}{9}\left[\begin{array}{c}
5 \\
10 \\
10
\end{array}\right] \in \operatorname{span} [ \left[\begin{array}{l}
1 \\
2 \\
2
\end{array}\right] ]
$$

또한, 이미 projection된 벡터에 대해 같은 projection matrix을 적용하면 달라지는 것이 없음도 확인할 수 있다. ($\boldsymbol{P}_{\pi}^{2} \boldsymbol{x} = \boldsymbol{P}_{\pi} \boldsymbol{x}$)
