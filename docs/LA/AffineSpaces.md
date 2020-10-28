# Affine Spaces

Affine 공간에 대해 엄밀하게 정의하기에 앞서 대략적인 느낌을 먼저 잡아보자.

<figure align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Affine_space_R3.png/1280px-Affine_space_R3.png" height=50% width=50% />
<figcaption>Wikipedia: Affine Space</figcaption>
</figure>

위 그림은 위키피디아에서 affine 공간을 설명하는 그림이다. 위의 3차원 공간에서 원점을 포함하는 평면 $P_1$이 있다. 그리고 이 $P_1$은 벡터부분공간이다. 하지만 $z$방향으로 평행이동된 평면 $P_2$는 원점을 포함하지 않을 뿐더러 $P_2$ 공간 내의 벡터 $\boldsymbol{a}, \boldsymbol{b}$의 합에 대해 닫혀있지도 않다. 따라서 $P_2$는 벡터부분공간이 아니고 이 공간을 affine 공간이라고 한다.

벡터공간에서 원점의 개념이 사라지므로 당황스럽지만 사실 우리가 살 수 있는 공간은 affine 공간의 하나라고 볼 수 있다. 이 세상의 원점은 존재하지 않기때문에 나의 위치와 친구의 위치를 더한 벡터는 정의될 수가 없다. 원점이 없으므로 두 벡터의 합이 정의될 수 없는 것이다. 하지만 벡터의 차이는 유효하다. 세상의 원점을 어디로 설정하더라도 **상대적**인 위치차이는 정의할 수 있으며 특정 점에 대한 벡터를 더하는 것은 가능하다. 이러한 느낌을 가지고 affine 공간에 대해 다루어보자.

## Affine Subspaces

> [!NOTE]
> **Definition: Affine Subspaces**
>
> 벡터공간 $V$에 대해 $\boldsymbol{x}_{0} \in V$이고, $U \subseteq V$가 부분공간이라면 다음의 부분집합을 $V$의 **affine subspace** 또는 **linear manifold**라고 한다.
> $$\begin{aligned} L &=\boldsymbol{x}_{0}+U:=\left\{\boldsymbol{x}_{0}+\boldsymbol{u}: \boldsymbol{u} \in U\right\} \\ &=\left\{\boldsymbol{v} \in V \mid \exists \boldsymbol{u} \in U: \boldsymbol{v}=\boldsymbol{x}_{0}+\boldsymbol{u}\right\} \subseteq V \end{aligned}$$
> 이 때, $U$는 **direction** 또는 **direction space**라고 하며 $\boldsymbol{x}_{0}$는 **support point**라고 한다.

Affine subspace는 $V$의 벡터부분공간이 아니라는 점에 유의하자.

설명을 보면 affine space가 특별한 경우인 것 같지만 사실 대부분의 공간은 affine subspace이다. 당장 3차원에서 생각할 수 있는 원점을 지나지 않는 점, 선, 면은 모두 affine subspace이다. 벡터공간에서 표현하자면 affine space형태의 공간을 support point만큼 평행이동시킨 공간으로 그려볼 수 있다.

선형대수학에서 기본이 되는 선형시스템의 해를 구하는 방법을 생각해보자. $\boldsymbol{Ax} = {b}$의 선형방정식을 풀 때, 일반해는 particular solution과 special solution의 합으로 나누어 풀게 되는데, 이 때 inhomogeneous system의 particular solution이 구성하는 공간이 바로 affine subspace가 되는 것이며 special solution이 구성하는 공간(homogeneous system)은 벡터부분공간이 된다.

## Affine Mappings

벡터공간에서의 선형변환처럼 affine subspace에서의 affine mapping에 대해 생각해 볼 수 있다. 두 mapping은 많은 성질을 공유한다. Affine mapping은 다음과 같이 정의된다.

> [!NOTE]
> **Definition: Affine Mapping**
>
> 두 벡터공간 $V$, $W$이 있을 때, 선형변환: $\Phi: V \rightarrow W$과 $W$의 원소 $\boldsymbol{a} \in W$에 대해 다음의 mapping을 $V$에서 $W$로의 **affine mapping**이라고 한다.
> $$\begin{aligned} \phi: V & \rightarrow W \\ & \boldsymbol{x} \mapsto \boldsymbol{a} + \Phi(\boldsymbol{x}) \end{aligned}$$
> 그리고 벡터 $\boldsymbol{a}$를 $\phi$의 **translation vector**라고 한다.

Affine mapping은 다음의 성질을 갖는다.

* 모든 affine mapping $\phi: V \rightarrow W$은 선형변환으로 구성할 수 있다. 예를 들어 $\Phi: V \rightarrow W$인 선형변환과 $W$ 내에서의 mapping $\tau: W \rightarrow W$가 있을 때, $\phi = \tau \circ \Phi$로 표현할 수 있으며 $\Phi$와 $\tau$는 유일하게 결정된다.
* Affine mapping끼리의 합성은 affine mapping이다.
* Affine mapping은 기하학적 구조를 보존한다.(차원, 평행 등)

## Conclusion

Affine 공간은 사실 새로운 개념이라기보다는 벡터공간을 공부하며 접했던 특정한 공간에 의미를 부여한 것에 가깝다. 책에서 언급하는 것처럼 머신러닝에서는 linear와 affine을 엄밀하게 구분하지는 않으므로 문맥에 따라 적절히 해석할 필요가 있다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
* [Wikipeida: Affine Space](https://en.wikipedia.org/wiki/Affine_space)