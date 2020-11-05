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

이제부터 벡터의 orthogonal projection을 inner product space ($\mathbb{R}^n, \langle \cdot, \cdot \rangle$)에서 벡터부분공간의 함수로 다루게 된다. 거창하게 표현되었지만 쉽게 표현하면 $n$차원 데이터와 내적이 정의된 공간을 벡터부분공간으로 보내는 함수에 관한 내용이다. 아래의 projection 내용을 다룰 때 별도의 언급이 없는 한 내적은 dot product $\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^\top \boldsymbol{y}$라고 가정한다.

### Projection onto One-Dimensional Subspaces (Lines)