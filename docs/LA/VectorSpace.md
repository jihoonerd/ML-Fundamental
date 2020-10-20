# Vector Space

## 1 Group

어떠한 집합 $\mathcal{G}$와 연산 $\otimes$에 대해 다음의 네가지 조건을 만족하면 군(Group)이라고 한다. 

$$G \coloneqq (\mathcal{G}, \otimes)$$

### 1.1 Closure of 

$\mathcal{G}$의 원소는 연산 $\otimes$에 대해 닫혀있다.

$$\forall x, y \in \mathcal{G} : x \otimes y \in \mathcal{G}$$

### 1.2 Associativity

$\mathcal{G}$의 원소는 결합특성을 만족한다.

$$
\forall x, y, z \in \mathcal{G} : (x \otimes y) \otimes z = x \otimes (y \otimes z)
$$

### 1.3 Neutral element

$\mathcal{G}$의 모든 원소 $x$에 대해 항등원 $e$가 존재한다.

$$
\exists e \in \mathcal{G} \ \forall x \in \mathcal{G} : x \otimes e = x \ \text{and} \ e \otimes x = x
$$

### 1.4 Inverse element

$\mathcal{G}$의 모든 원소 $x$에 대해 역원 $y$가 존재한다.

$$
\forall x \in \mathcal{G} \ \exists y \in \mathcal{G} : x \otimes y = e \ \text{and} \ y \otimes x  = e
$$


## 2 Abelian Group

특히, 위에 언급된 네 가지 조건과 더불어 $\otimes$연산에 대해 다음 조건을 추가로 만족하면 해당 군 $G = (\mathcal{G}, \otimes)$를 아벨군(Abelian Group) 또는 가환군(Commutative Group)이라고 한다.

$$
\forall x, y \in \mathcal{G} : x \otimes y = y \otimes x
$$

# 3 Vector Space

학부 선형대수학 수업시간을 돌이켜보면 벡터공간을 정의할 때 위와 같은 군(Group)에 대한 정의가 아닌 개별 성질에 대해 바로 언급을 했었던 것으로 기억한다. 그러나 군을 이용하면 훨씬 간결하게 벡터공간을 정의할 수 있다.

실수벡터공간(Real-valued Vector Space) $V = (\mathcal{V}, +, \cdot)$은 집합 $\mathcal{V}$에 대한 다음 두 연산

$$
\+ : \mathcal{V} \times \mathcal{V} \to \mathcal{V}
$$

$$
\cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}
$$

에 대해 다음 조건을 만족하는 군이다.

1. $(\mathcal{V}, +)$는 아벨군이다.
2. $(\mathcal{V}, +, \cdot)$은 다음성질을 만족한다.
   1. 분배법칙(Distributivity):
      1. $\forall \lambda \in \mathbb{R}, \ \boldsymbol{x}, \boldsymbol{y} \in \mathcal{V} : \lambda \cdot(\boldsymbol{x} + \boldsymbol{y}) = \lambda \cdot \boldsymbol{x} + \lambda \cdot \boldsymbol{y}$
      2. $\forall \lambda, \psi \in \mathbb{R}, \ \boldsymbol{x} \in \mathcal{V} : (\lambda + \psi) \cdot \boldsymbol{x} = \lambda \cdot \boldsymbol{x} + \psi \cdot \boldsymbol{x}$
   2. 결합법칙(Associativity): $\forall \lambda, \psi \in \mathbb{R}, \ \boldsymbol{x} \in \mathcal{V} : \lambda \cdot (\psi \cdot \boldsymbol{x}) = (\lambda \psi) \cdot \boldsymbol{x}$
   3. 항등원(Neutral element): $\forall \boldsymbol{x} \in \mathcal{V} : 1 \cdot \boldsymbol{x} = \boldsymbol{x}$

그리고 위의 벡터공간 의 원소 $\boldsymbol{x} \in V$를 벡터(vector)라고 한다.

## 3.1 Vector Subspace

벡터부분공간(Vector Subspaces)는 해당 공간의 원소에 대한 연산의 결과가 원래 공간을 벗어나지 않는 공간이다.

엄밀한 정의는 다음과 같다.

$V = (\mathcal{V}, +, \cdot)$이 벡터공간이고 $\mathcal{U}	\subseteq \mathcal{V}, \mathcal{U} \neq 0$일때, 다음을 만족하는 $U = (\mathcal{U}, +, \cdot)$를 $V$의 벡터부분공간이라고 한다.

* 만약 $U$가 연산 $ + (\mathcal{U} \times \mathcal{U}) $, $ \cdot (\mathbb{R} \times \mathcal{U}) $에 대한 벡터공간이면 $U \subseteq V$라고 쓰며 $U$를 $V$의 벡터부분공간이라 한다.

벡터 부분공간 $ U = (\mathcal{U}, +, \cdot) $ 역시 벡터공간이므로 덧셈에 대한 아벨군, 그리고 덧셈, 곱셈에 대한 Distributivity, Associativity, Neutral element를 갖는 것은 물론, 다음의 성질을 추가로 만족해야한다.

1. $\mathcal{U} \neq \emptyset$, in particular: $\boldsymbol{0} \in \mathcal{U}$
2. Closure of $U$
   1. $\forall \lambda \in \mathbb{R} \ \forall {\boldsymbol{x}} \in \mathcal{U} : \lambda \boldsymbol{x} \in \mathcal{U}$
   2. $\forall \boldsymbol{x}, \boldsymbol{y} \in \mathcal{U} : \boldsymbol{x} + \boldsymbol{y} \in \mathcal{U}$

# 4 Conclusion

기계학습의 이론적 배경이 되는 벡터공간의 엄밀한 정의에 대해서 살펴보았다. 정성적으로 요약하면 공간 내 원소간의 합이나 원소와 스칼라곱에 대해 정의가 되는 공간이며 이는 선형대수학에서 선형성을 바탕으로 이론을 발전시킬 수 있는 무대를 제공한다.

# 5 Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.