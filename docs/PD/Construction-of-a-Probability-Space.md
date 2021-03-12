# Construction of a Probability Space

확률이론에서는 어떤 실험에 대한 random outcome에 대한 수학적인 구조를 정의하는 것을 목표로 한다.

## Philosophical Issues

확률을 다루는 중요한 이유는 logical reasoning을 하기 위해서이다. 인생사가 그렇듯 거의 모든 일들은 옳고/그름, 예/아니오로 간단하게 정의되지 않는다. 이렇게 이분법적으로 보는 방식을 Boolean logic이라고하는데 우리에게 필요한 것은 신빙성, 가능성, 불확실성과 같이 0과 1사이의 값을 다룰 수학적인 체계이다.

철학적으로 이에대한 연구가 있었고 E.T. Jaynes는 가능성(plausibility)이라는 성질을 다루기 위해서는 다음 세개의 수학적 기준이 필요함을 확인하였다.

1. 가능성은 실수로 표현되어야 한다.
2. 이 숫자들은 상식에 부합하는 규칙을 토대로 한다.
3. 이를 통해 얻는 추론은 일관성(consistency)이 있어야 한다. 일관성은 다음과 같이 정의한다.
   1. Consistency or non-contradiction: 다른 방식을 통해서 동일한 결과를 얻을 수 있다면 이러한 모든 방식들에 대해서도 같은 가능성이 부여된다.
   2. Honesty: 모든 데이터는 설명이 가능해야 한다.
   3. Reproducibility: 두 가지 문제에 대해 같은 지식을 갖고 있다면 같은 가능성을 부여한다.

이 Cox-Jaynes theorem에서 정의한 가능성에 대한 정의로 가능성 $p$와 이에 대한 임의의 단조함수에 의한 변환까지도 수학적으로 충분히 다룰 수 있음을 보였다.

## Probability and Random Variable

교재에서는 확률과 관련하여 다음의 세가지 구분되는 개념이 혼용되며 사용되고 있다고 말한다.

1. Probability Space
2. Random Variable
3. Distribution / Law associated with a random variable

따라서 확률과 분포를 다루는 문서의 목표 중 하나는 위 세가지 개념을 명확히 구분하는 것이다.

우선 확률을 다루는 공간에 대해 살펴보자. 현대 수학에서 사용하는 확률의 개념은 Kolmogorov가 제시한 **sample space, event space, probability measure**, 세 가지 공리에 기초하고 있다. 확률공간(Probability Space)은  random outcome을 만들어내는 real-world process에 대해 다룬다.

* Sample Space $\Omega$
  Sample space는 시행을 통해서 얻을 수 있는 모든 가능한 결과(outcome)의 집합니다. 예를들어 동전을 두 번 던지는 시행을 하였다면 sample space는 $\{hh, tt, ht, th\}$로 네개가 된다.

* Event Space $\mathcal{A}$
  Event space는 시행을 통해서 얻을 수 있는 가능한 결과의 공간이다. 따라서 event space는 sample space의 subset이 된다. 

* Probability $P$
  Event $A \in \mathcal{A}$에 대해서 확률, 또는 degree of belief에 대한 measure로써 $P(A)$를 정의하며 이를 probability of $A$라고 한다.

하나의 event에 대한 확률은 $[0, 1]$의 값을 가지며 sample space $\Omega$의 모든 outcome에 대한 확률의 합은 반드시 1이 되어야 한다.
$$P(\Omega) = 1$$
확률 공간 $(\Omega, \mathcal{A}, P)$가 주어졌을 떄 우리는 실제 세계의 현상을 설명하기 위해 확률공간을 사용하게 된다.

머신러닝에서는 명시적으로 확률공간을 직접 다루지는 않고 대신 우리가 관심있는 확률공간인 target space $\mathcal{T}$로써 다룬다. 그렇다면 sample space에서 target space로의 mapping에 대해서도 알아야 한다.
$$X: \Omega \righatarrow \mathcal{T}$$
Sample space의 outcome에서 관심있는 target space로의 mapping을 **random variable**이라고 한다. 예를 들어 두 개의 동전을 던진 결과로써 나올 수 있는 outcome은 hh, ht, th, tt이며 관심의 대상($\mathcal{T}$)이 앞면의 나온 횟수라면 각각의 outcome에 대한 random variable은 다음과 같이 mapping된다.
$$\begin{aligned}
X(hh) = 2 \\ X(ht) = 1 \\ X(th) = 1 \\ X(tt) = 0
\end{aligned}$$
Target space $\mathcal{T}$는 $\mathcal{T} = \{0, 1, 2\}$이고 $\mathcal{T}$의 원소의 확률, 즉 앞면이 나오지 않거나, 한 번, 또는 두 번 나올 확률이 우리가 관심있는 대상인 것이다. Sample space $\Omega$와 target space $\mathcal{T}$가 유한집합이라면 random variable함수는 lookup table이 된다. Target space의 부분집합 $S \subseteq \mathcal{T}$에 대해서 random variable $X$에 대한 특정 사건($S$)이 일어날 확률을 엄밀하게 $P_{X}(S) \in [0, 1]$로 표현한다.
