# Linear Independence

벡터공간은 앞으로 다루게 될 벡터들이 속한 공간이 어떻게 정의되는지, 어떤 성질을 갖는지를 정의했다. 선형대수학은 그 이름에서 드러나듯 선형성을 토대로 이론을 발전시킨다. 따라서 여기서는 선형결합과 선형독립이라는 두 가지 주제를 다룬다. 특히, 선형독립은 이후에 다룰 랭크나 내적등의 개념을 이야기할 때 필수적인 개념이므로 정확하게 이해할 필요가 있다.

## Basis

벡터공간에 속하는 벡터는 정의에 의해 스칼라만큼 곱하거나 벡터끼리 더해도 Closure property에 의해 같은 벡터공간내 다른 벡터로써 표현된다. 그렇다면 어떤 벡터공간이 있을 때, 특정 벡터들을 스칼라 곱을 하거나 더해 공간의 모든 벡터를 표현할 수 있는 벡터집합을 생각해 볼 수 있다. 이러한 벡터들의 집합을 **기저(basis)** 라고 한다.

예를 들어, 2차원의 대표적인 기저가 $\left\{ \begin{bmatrix}1 \\ 0\end{bmatrix}, \begin{bmatrix}0 \\ 1\end{bmatrix} \right\}$, 3차원의 대표적인 기저가 $\left\{ \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix}, \begin{bmatrix}0 \\ 1 \\ 0\end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \right\}$이다. 

<div align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/3d_two_bases_same_vector.svg/130px-3d_two_bases_same_vector.svg.png"/>
</div>

3차원 벡터공간의 임의의 벡터는 위의 기저벡터들의 스칼라곱과 합으로써 표현이 가능하다.

## Linear Combination

선형결합의 수학적 정의는 다음과 같다.

> [!NOTE]
> **Definition: Linear Combination**
>
> 어떤 벡터공간 $V$와 이 공간에 속하는 유한개의 벡터 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k} \in V$에 대해 다음과 같은 꼴로 표현되는 모든 $\boldsymbol{v} \in V$를 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}$의 **선형결합(linear combination)** 이라고 한다.
> $$\boldsymbol{v} = \lambda_{1} \boldsymbol{x}_{1} + \cdots + \lambda_{k} \boldsymbol{x}_{k} = \sum_{i=1}^{k} \lambda_{i} \boldsymbol{x}_{i} \in V \quad (\lambda_{1}, \ldots, \lambda_{k} \in \mathbb{R})$$

이 때 모든 계수 $\lambda_{i} = 0$이면 $\boldsymbol{0}$이 되어 영벡터 $\boldsymbol{0}$는 trivial linear combination이라고 한다. 모든 계수가 0이 아니면서 선형결합이 영벡터가 되는 형태를 non-trivial linear combination이라고 한다.

## Linear Independence

선형결합을 정의했으므로 이제 선형독립을 간결하게 정의할 수 있게된다. 선형독립은 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Linear Independence**
>
> 벡터공간 $V$와 $k \in \mathbb{N}$인 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k} \in V$에 대해 non-trivial solution이 존재하면, 즉 $\boldsymbol{0} = \sum_{i=1}^{k} \lambda_{i} \boldsymbol{x}_{i}$를 만족하는 $\lambda_{i} \neq 0$가 적어도 한 개 이상 존재하면 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}$는 **선형종속(linearly dependent)** 이라고 하며, 오직 trivial solution $(\lambda_{1} = \cdots = \lambda_{k} = 0)$ 만이 존재한다면 이 때의 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}$를 **선형독립(linearly independent)** 이라고 한다.

선형독립은 선형대수학에서 가장 중요한 개념중에 하나로 선형독립관계에 있는 벡터들은 중복되는(redundance)정보가 없다고 볼 수 있다. 따라서 선형독립으로 구성된 벡터들에서 벡터를 제외하게 된다면 제외하기 전 선형독립 벡터들이 구성하고 있던 벡터공간을 표현할 수 없게 된다.

### Properties of Linearly Independent Vectors

다음은 선형독립인 벡터들이 갖는 성질들이다.

* $k$개의 벡터가 있다고 할 때, 벡터들은 선형종속 혹은 선형독립 둘 중 한가지이다. 다른 경우는 없다.
* 어느 한 벡터가 0이거나 두 벡터가 같으면 해당 벡터들은 선형종속이다.
* 한 벡터가 다른 벡터들의 선형결합으로 구성된다면 해당 벡ㅌ터들은 선형종속이다.
* 가우스 소거법(Gaussian elimination)을 하여 row echelon form을 만들었을 때, 모든 열벡터가 pivot일 때만 선형독립이다. 하나라도 non-pivot인 열벡터가 있다면 선형종속이다.

## Conclusion

이번 문서에서는 기저와 선형결합, 선형독립에 대해 알아보았다. 간단한 개념이지만 이 개념들은 이후 선형대수학 내용의 기초가 되므로 정확하게 이해할 필요가 있다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
* [Wikipedia: Basis](https://en.wikipedia.org/wiki/Basis_(linear_algebra))