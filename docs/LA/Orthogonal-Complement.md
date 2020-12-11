# Orthogonal Complement

앞서 두 벡터의 직교에 대해 다루었다. 이를 확장해 보면 벡터가 아닌 두 평면이 직교하고 있는 공간을 생각해보자. 방의 모서리 부분을 보면 세 평면이 직교하는 것을 볼 수 있다. 각각의 평면을 벡터공간으로 정의하면 직교하는 벡터공간이 된다. 공간의 직교개념은 기학적 관점에서 차원축소를 이해하는데 중요한 개념이 된다. 이 문서에서는 벡터공간의 직교에 대해 알아보자.

## Orthogonal Complement

Orthogonal complement는 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Orthogonal Complement**
>
> $D$-차원의 벡터공간 $V$와 $M$-차원의 벡터부분공간 $U \subseteq V$가 있다고 가정하자. 이 때 Orthogoanl complement $U^\perp$은 $D-M$-차원으로 $V$의 벡터부분공간이다. 그리고 이 벡터공간 $U^\perp$는 벡터공간 $V$에서 $U$와 직교하는 모든 벡터를 포함하게 된다. 또한, $U \cap U^\perp = \{\boldsymbol{0}\}$로 $\boldsymbol{0}$만 두 벡터공간에 공통으로 속한다.
>
> 따라서, 벡터공간 $V$의 임의의 벡터 $\boldsymbol{x} \in V$는 $U$의 basis $(\boldsymbol{b}_1, \ldots, \boldsymbol{b}_M)$와 $U^\perp$의 basis $(\boldsymbol{b}_1^\perp, \ldots, \boldsymbol{b}_{D-M}^\perp)$에 대해 다음과 같이 유일하게 분해될 수 있다.
> $$\boldsymbol{x} = \sum_{m=1}^M \lambda_m \boldsymbol{b}_m + \sum_{j=1}^{D-M} \psi_j \boldsymbol{b}_j^\perp, \quad \lambda_m, \psi_j \in \mathbb{R}$$

벡터공간에서 기하적으로 어떻게 표현되는지를 살펴보자. Orthogonal complement는 다음 그림에서 2차원 평면 $U$를 표현하는데 사용될 수 있다. 다음의 그림은 이러한 상황을 표현한다.

<figure align=center>
<img src="assets/images/LA/Fig_3.7.png" width=30% height=30%/>
<figcaption>Figure 3.7</figcaption>
</figure>

위 그림에서 $\boldsymbol{w}$는 $U$에 속하는 모든 벡터들에 대해 수직이며 이러한 벡터 $\boldsymbol{w}$를 $U$의 **normal vector**라고 한다.

일반적으로 orthogonal complement는 벡터공간이나 affine 공간에서 hyperplane을 표현하는데 사용될 수 있다.

## Conclusion

Orthogonal complement는 앞에서 다룬 벡터의 직교를 직교하는 벡터공간으로 확장한 것이다. 3차원공간에서 생각해보면 orthogonal complement 개념을 직관적으로 이해하기 편한다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
