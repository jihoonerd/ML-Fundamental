# Vector Space

여기서는 선형대수학의 무대인 벡터공간과 벡터부분공간을 정의한다.

학부과정에서 벡터공간은 이런저런 성질들을 만족하는 공간으로 학습했었던 기억이 있다. 하지만 돌이켜보면 그 성질들은 결국 군(group)의 성질을 풀어쓴 것이었다. 따라서 군을 먼저 정리하고 군을 이용하게 깔끔하게 벡터공간을 정의하는 것이 이 문서의 목표이다.

## Group

너무 깊게 들어갈 필요는 없지만 대수구조에 대해 간단하게 알아보자. 수학에는 추상대수학(abstract algebra)분야가 있으며 이 분야에서는 대수 구조를 다룬다. 대수 구조는 연산의 집합으로 생각하면 된다. 그리고 이러한 대수 구조를 이루는 것 중 하나가 바로 군이다.

아래 그림은 대수 구조를 나타낸다.

<figure align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/d/d7/Algebraic_structures.png"/>
<figcaption>Wikipedia: Algebraic structures</figcaption>
</figure>

도표는 마치 프로그램의 상속구조처럼 읽으면 된다. 예를 들어 가장 상단을 볼 때 Magma는 Set의 성질을 상속받고 이항연산(bin. op.)에 대해 닫혀있는 성질을 추가로 갖는 것이다. 정말 무시무시하게 생긴 도표지만 관심있는 것은 어차피 군이므로 오른쪽 상단에 위치한 군(Group)을 보자.

> [!NOTE]
> Group: Monoid w/ inverses

라고 씌어있다. 역원을 갖는 모노이드라는 뜻이다. 모노이드는 결합법칙을 만족하고(Semigroup) 항등원을 갖춘 대수 구조이다. 또한 Magma의 성질도 상속하므로 이항연산에 대해 닫혀있음도 알 수 있다. 이제 군을 좀 더 이해하기 쉬운 형태로 쓰면 어떠한 이항 연산에 대해 결합법칙을 만족하고 항등원, 역원을 갖는 닫혀있는 대수 구조라고 말할 수 있다.

다시 말해, 어떤 집합 $\mathcal{G}$와 연산 $\otimes$에 대한 군 $G$은 다음과 같이 표기한다.

$$G \coloneqq (\mathcal{G}, \otimes)$$

군은 다음 공리를 만족한다.

* Closure
* Associativity
* Identity element
* Inverse element

하나씩 정의를 살펴보자.

### Closure

$\mathcal{G}$의 원소는 연산 $\otimes$에 대해 닫혀있다.

$$\forall x, y \in \mathcal{G} : x \otimes y \in \mathcal{G}$$

### Associativity

$\mathcal{G}$의 원소는 결합법칙을 만족한다.

$$\forall x, y, z \in \mathcal{G} : (x \otimes y) \otimes z = x \otimes (y \otimes z)$$

### Identity element

$\mathcal{G}$의 모든 원소 $x$에 대해 항등원 $e$가 존재한다.

$$\exists e \in \mathcal{G} \ \forall x \in \mathcal{G} : x \otimes e = x \ \text{and} \ e \otimes x = x$$

### Inverse element

$\mathcal{G}$의 모든 원소 $x$에 대해 역원 $y$가 존재한다.

$$
\forall x \in \mathcal{G} \ \exists y \in \mathcal{G} : x \otimes y = e \ \text{and} \ y \otimes x  = e
$$


## Abelian Group

특히, 위에 언급된 네 가지 조건과 더불어 $\otimes$연산에 대해 다음 조건을 추가로 만족하면 해당 군 $G = (\mathcal{G}, \otimes)$를 아벨군(Abelian Group) 또는 가환군(Commutative Group)이라고 한다.

$$
\forall x, y \in \mathcal{G} : x \otimes y = y \otimes x
$$

## Vector Space

이제 벡터공간을 깔끔하게 정의하기위한 군에 대한 개념은 충분하다. 앞서 살펴본 군을 사용해 벡터공간을 정의해보자.

실수벡터공간(Real-valued Vector Space) $V = (\mathcal{V}, +, \cdot)$은 집합 $\mathcal{V}$에 대한 다음 두 연산

$$
+ : \mathcal{V} \times \mathcal{V} \to \mathcal{V}
$$

$$
\cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}
$$

에 대해 다음 조건을 만족하는 군이다.

> [!NOTE]
> **Definition: Vector Space**
>
> 1. $(\mathcal{V}, +)$는 아벨군(Abelian Group)이다.
> 2. $(\mathcal{V}, +, \cdot)$은 다음성질을 만족한다.
>    1. 분배법칙(Distributivity):
>       1. $\forall \lambda \in \mathbb{R}, \ \boldsymbol{x}, \boldsymbol{y} \in \mathcal{V} : \lambda \cdot(\boldsymbol{x} + \boldsymbol{y}) = \lambda \cdot \boldsymbol{x} + \lambda \cdot \boldsymbol{y}$
>       2. $\forall \lambda, \psi \in \mathbb{R}, \ \boldsymbol{x} \in \mathcal{V} : (\lambda + \psi) \cdot \boldsymbol{x} = \lambda \cdot \boldsymbol{x} + \psi \cdot \boldsymbol{x}$
>    2. 결합법칙(Associativity): $\forall \lambda, \psi \in \mathbb{R}, \ \boldsymbol{x} \in \mathcal{V} : \lambda \cdot (\psi \cdot \boldsymbol{x}) = (\lambda \psi) \cdot \boldsymbol{x}$
>    3. 항등원(Neutral element): $\forall \boldsymbol{x} \in \mathcal{V} : 1 \cdot \boldsymbol{x} = \boldsymbol{x}$

그리고 위의 벡터공간 의 원소 $\boldsymbol{x} \in V$를 벡터(vector)라고 하며 덧셈에 대한 항등원을 $\boldsymbol{0}$, 영벡터(zero vector)라고 한다. 이런저런 다양한 성질을 만족해야 할 것 같지만, 수학적으로 위의 조건만 만족하면 벡터공간을 구성하게 된다.

### Vector Subspace

벡터부분공간(Vector Subspaces)는 해당 공간의 원소에 대한 연산의 결과가 원래 공간을 벗어나지 않는 공간이다.

엄밀한 정의는 다음과 같다.

> [!NOTE]
> **Definition: Vector Subspace**
>
> $V = (\mathcal{V}, +, \cdot)$이 벡터공간이고 $\mathcal{U} \subseteq \mathcal{V}, \mathcal{U} \neq \emptyset$일때, $U = (\mathcal{U}, +, \cdot)$가 연산 $ + (\mathcal{U} \times \mathcal{U}) $, $ \cdot (\mathbb{R} \times \mathcal{U})$에 대한 벡터공간이면 $U \subseteq V$라고 쓰며 $U$를 $V$의 벡터부분공간이라 한다.

벡터 부분공간 $ U = (\mathcal{U}, +, \cdot) $ 역시 벡터공간이므로 벡터공간의 성질을 모두 만족해야하며 다음의 성질을 추가적으로 만족해야 한다.

1. $\mathcal{U} \neq \emptyset$, in particular: $\boldsymbol{0} \in \mathcal{U}$
2. Closure of $U$
   1. $\forall \lambda \in \mathbb{R} \ \forall {\boldsymbol{x}} \in \mathcal{U} : \lambda \boldsymbol{x} \in \mathcal{U}$
   2. $\forall \boldsymbol{x}, \boldsymbol{y} \in \mathcal{U} : \boldsymbol{x} + \boldsymbol{y} \in \mathcal{U}$

## Conclusion

머신러닝의 이론적 배경이 되는 벡터공간을 정의하기 위해 군에 대해 간단히 알아보고 벡터공간, 벡터부분공간의 엄밀한 정의에 대해서 살펴보았다. 정성적으로 요약하면 공간 내 원소간의 합이나 원소와 스칼라곱에 대해 정의가 되는 공간이며 이는 선형대수학에서 이론을 발전시킬 수 있는 무대를 제공한다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
* [Wikipedia: Vector Space](https://en.wikipedia.org/wiki/Vector_space)