# Basis

앞의 [선형독립문서](/LA/LinearIndependence.md)에서 기저(basis)의 정의를 간단하게 다루었다. 여기서는 geterating set과 span을 사용해 기저와 기저의 성질을 다룬다. 그리고 차원의 개념을 기저로써 정의해본다.

## Generating Set and Span

Generating set은 문자 그대로 풀어내면 "만들어내는 집합"이라는 뜻이다. 어떤 벡터집합을 가지고 있다면 이 벡터들의 선형결합을 통해 벡터공간을 만들 수 있을 것이다. 이런 벡터공간을 만드는데 재료역할을 하는 벡터집합을 generating set이라고하며 만들어진 벡터공간은 span이라고 한다. 엄밀한 정의는 아래와 같다.

> [!NOTE]
> **Definition: Generating Set**
>
> 벡터공간은 덧셈과 곱셈에 대한 특정 조건들을 만족하는 군이다. 따라서 벡터공간은 $V = (\mathcal{V}, +, \cdot)$으로 표기하고 이 벡터공간의 벡터 집합 $\mathcal{A}$를 $\mathcal{A} = \{ \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k} \} \subseteq	\mathcal{V}$라고 하자. 이 때, $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}$의 선형결합으로 표현가능한 모든 벡터 $\boldsymbol{v} \in \mathcal{V}$에 대해서 $\mathcal{A}$를  **generating set** of $V$라고 한다.

> [!NOTE]
> **Definition: Span**
>
> $\mathcal{A}$의 벡터 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}$로 만들어지는 선형결합의 집합을 **span** of $\mathcal{A}$라고 한다. $\mathcal{A}$가 벡터공간 $V$로 span한다면 $V = \text{span}[\mathcal{A}]$ 또는 $V = \text{span}[\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{k}]$으로 표기한다.

## Definition of Basis

이제 generating set과 span을 사용해 기저를 정의해보자. Generating set은 꼭 서로 선형독립으로 중복되는 정보가 없을 필요는 없다. 예로 2차원 평면의 generating set이 꼭 $\left\{ \begin{bmatrix}1 \\ 0\end{bmatrix}, \begin{bmatrix}0 \\ 1\end{bmatrix} \right\}$일 필요는 없다. $\left\{ \begin{bmatrix}1 \\ 0\end{bmatrix}, \begin{bmatrix}0 \\ 1\end{bmatrix}, \begin{bmatrix}1 \\ 1 \end{bmatrix} \right\}$일 수도 있는 것이다. 따라서 generating set의 정의에 조건을 추가해 기저를 정의할 수 있다. 기저는 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Basis**
>
> 어떤 벡터공간 $V = (\mathcal{V}, +, \cdot)$과 벡터집합 $\mathcal{A} \subseteq \mathcal{V}$가 있다고 해보자. $V$로 span하는 generating set $\mathcal{A}$보다 작은 $\tilde{\mathcal{A}}$가 존재하지 않는다면 $(\tilde{\mathcal{A}} \subsetneq \mathcal{A} \subseteq \mathcal{V})$ $\mathcal{A}$는 $V$의 **minimal** generating set이라고 한다. 그리고 $V$에 대해서 선형 독립인 minimal generating set을 $V$의 **기저(basis)** 라고 한다.

## Properties of Basis

벡터공간 $V = (\mathcal{V}, +, \cdot)$와 공집합이 아닌 $\mathcal{B} \subseteq \mathcal{V}$에 대해서 다음의 표현은 동치이다.

* $\mathcal{B}$는 $V$의 기저이다.
* $\mathcal{B}$는 $V$의 minimal generating set이다.
* $\mathcal{B}$는 $V$에 대해 가장 큰 선형독립벡터의 집합이다. 이 집합에 어떠한 벡터가 더해지면 선형종속이 된다.
* $V$에 속하는 모든 벡터 $\boldsymbol{x} \in V$는 $\mathcal{B}$의 선형결합으로 표기할 수 있으며 모든 선형결합은 유일(unique)하다.

## Dimension

어떤 공간을 이야기할 때 우리는 차원(dimension)이라는 단어를 사용한다. 기저를 사용해 차원을 정의하면 다음과 같다.

> [!NOTE]
> **Definition: Dimension**
>
> 벡터공간 $V$의 차원은 $V$의 기저벡터의 수와 같다. 그리고 이를 $\text{dim}(V)$로 표기한다. $V$의 벡터부분공간 $U \subseteq V$는 $\text{dim}(U) \leqslant \text{dim}(V)$가 성립하며 등호는 벡터공간 $U = V$일때만 성립한다.

> [!Warning]
> 차원과 벡터를 구성하는 원소의 수는 무관하다. 예를 들어 $V = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$이면 벡터를 구성하는 원소는 2개이지만 차원은 1차원이다.

## Conclusion

이번 문서에서는 기저를 중심으로 다루었다. 선형대수학은 벡터공간에서의 변환을 다루게 되는데 벡터공간에서의 기준역할을 해주는 것이 바로 기저이다. 2차원이나 3차원에서 익숙하게 사용하는 좌표계도 암묵적으로 합의한 기저가 있기 때문에 공간내 벡터를 표현하는 수 많은 방식이 있음에도 혼동없이 사용할 수가 있는 것이다. 

<div align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/AI_aircraft_orientation.png" width=50% height=50%/>
</div>

전투기를 조종하는 파일럿은 야간음속비행에서 바다와 하늘을 구분하기 어려운 경우가 있다고 한다. "나"를 기준으로 위쪽 방향이 늘 하늘은 아닌 것이기 때문에 기준이 되는 벡터가 필요하고, 조종사에게 계기판(HUD)의 자세지시계(Attitude indicataor)는 일반적으로 땅에 있을 때 기준의 하늘방향벡터를 제시해 줌으로써 의도한 방향으로 갈 수 있는 것과 비슷한 이치라고 보면 되겠다.

기저는 이후에 다루는 개념에서 유기적으로 사용되므로 명확하게 이해하도록 하자.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
* [Wikipedia: Attitude indicator](https://en.wikipedia.org/wiki/Attitude_indicator)