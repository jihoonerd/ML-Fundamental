# Inner Product of Functions

## Vector with Infinite Number of Entries

지금까지 내적은 유한차원의 벡터에 대한 내적으로 길이, 거리, 각도를 정의할 때 사용하였다. 유한차원의 벡터는 각각의 벡터가 $\boldsymbol{x} \in \mathbb{R}^{n}$로 $n$개의 유한한 값을 갖는 벡터이다. 이 개념을 무한개의 값을 갖는 벡터로 확장하는 것도 가능하다.

참고로, 무한하다는 것은 두 가지 종류로 나뉘는데 셀 수 있는 무한(countably infinite)과 셀 수 없는 무한(uncountably infinite)이다. 자연수의 집합으로 다루고 있는 무한대의 집합이 각각 대응을 시킬 수 있으면 이를 셀 수 있는 무한, 또는 denumerable이라고 한다. 이러한 집합의 크기(cardinality)는 자연수의 갯수와 같으며 유리수집합이 이에 해당한다. 반면, 실수 집합과 같이 자연수의 집합으로 대응시킬 수 없는 집합을 셀 수 없는 무한이라고 한다.

지금 다루는 유한차원의 벡터를 무한대로 확장하는 것은 셀 수 있는 무한, 셀 수 없는 무한 모두에 적용할 수 있다.

## Inner Product of Two Functions

두 함수, $u: \mathbb{R} \rightarrow \mathbb{R}$, $v: \mathbb{R} \rightarrow \mathbb{R}$의 내적은 
다음과 같은 정적분(definite integral)로 정의된다.

$$\langle u, v \rangle \coloneqq \int_{a}^{b} u(x)v(x)dx$$

이 때 lower, upper limit은 $a, b < \infty$를 만족해야한다.

함수의 내적정의를 통해 앞서 사용한 norm이나 직교의 개념을 함수에 적용할 수 있다. 만약 함수의 내적이 0이라면 두 함 수 $u, v$는 직교한다고한다. 하지만 유한차원 벡터의 내적과는 다르게 함수의 내적은 무한대로 발산할 수 있다는 특징도 있다. 함수의 내적을 보다 엄밀히 정의하기 위해서는 함수해석학과 Hilbert 공간을 다루어야 하나, 이는 책에서 다루고자 하는 범주를 벗어나므로 다루지 않는다.

## Example

삼각함수인 $\sin(x)$와 $\cos(x)$를 곱한 함수인 $\sin(x) \cos(x)$는 $f(-x) = -f(x)$를 만족하는 함수로 함수의 내적 정의에 따라 $a=-\infty$, $b=\infty$범위에서 내적하면 0이된다. 따라서 $\sin$과 $\cos$은 직교하는 함수이다.

이를 확장하면 매우 재미있는 사실을 알 수 있게된다. 같은 원리로 다음과 같은 $\cos$ 함수의 집합을 생각해보자.
$$\{1, \cos(x), \cos(2x), \cos(3x), \cdots\}$$
이 집합에 있는 함수들은 모두 직교하는 함수이다. 이러한 함수들을 span시키면 $[-\pi, \pi)$범위에서 even하고 주기함수의 성질을 갖는 다양한 함수들을 만들 수 있을 것이다. 또한 이쯤에서 위의 집합은 지금까지 다룬 basis와 basis가 span하는 공간과 같은 역할을 하고 있다는 것을 눈치챌 수 있을 것이다.

이를 반대로 생각해보자. $[-\pi, \pi)$범위에서 even하고 주기함수의 성질을 갖는 임의의 함수는 위 집합으로 구성할 수 있다는 뜻이 된다. 이러한 함수를 위의 basis역할을 하는 주기함수의 집합으로 projection하는 것이 바로 Fourier series의 기본적인 아이디어이다.

## Conclusion

이 문서에서는 함수의 내적에 대해 알아보았다. 내적의 개념은 벡터뿐만이 아니라 함수, 심지어 이후 다룰 random variable에 대해서도 폭넓게 적용가능하다는 사실을 알아두자.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.