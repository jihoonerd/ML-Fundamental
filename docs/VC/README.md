# Vector Calculus

머신러닝에서의 문제들은 목적함수 혹은 비용함수를 최적화함으로써 풀게된다. 구체적으로는 최적화의 대상은 parameter이며 목적을 달성하기 위한 최적의 parameter를 찾아내는 것이 목표이다. 대부분의 경우 이러한 최적화 문제는 gradient기반의 방법으로 풀게되는데 이 때 필수적으로 사용하게 되는 것이 vector calculus이다.

Vector calculus에서는 함수를 다루며 함수 $f$는 input $\boldsymbol{x} \in \mathbb{R}^{D}$와 target인 $f(\boldsymbol{x})$를 이어주는 역할을 한다. 이 때 input의 공간을 $f$의 domain, $f(\boldsymbol{x})$의 공간을 $f$의 image/codomain이라고 한다.

함수의 domain, codomain의 관계를 표한하기 위해 다음과 같은 표기를 주로 사용하게 된다.

$$
\begin{aligned}
f: \mathbb{R}^{D} & \rightarrow \mathbb{R} \\
\boldsymbol{x} & \mapsto f(\boldsymbol{x})
\end{aligned}
$$

Vector calculus에서는 함수들의 gradient를 어떻게 계산하는지를 주로 다루게 된다.