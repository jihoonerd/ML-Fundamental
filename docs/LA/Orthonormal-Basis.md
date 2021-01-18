# Orthonormal Basis

Orthonormal basis는 이름에서 나타내듯 basis vector가 서로 직교하고 크기가 각각 1인 경우이다. 이번 문서에서는 orthonormal basis에 대해 다룬다.

우선, orthonormal bassis를 정의해보자.

> [!NOTE]
> **Definition: Orthonormal Basis**
>
> $n$-차원 벡터공간 $V$와 $V$의 basis $\{\boldsymbol{b}_1, \ldots, \boldsymbol{b}_n\}$에 대해 모든 $i, j = 1, \ldots, n$가 다음을 만족하면 **Orthonormal Basis(ONB)** 라고 한다.
> $$\begin{aligned} \langle \boldsymbol{b}_i, \boldsymbol{b}_j \rangle &= 0 \quad \operatorname{for} \ i \neq j (i, j \in \mathbb{N}) \\ \langle \boldsymbol{b}_i, \boldsymbol{b}_i \rangle &= 1 \end{aligned}$$
> $\langle \boldsymbol{b}_i, \boldsymbol{b}_j \rangle = 0$ 조건만 만족하는 경우에는 **orthogonal basis** 라고 한다.


## Computing Orthonormal Basis

가우스 소거법(Gaussian elimination)을 통해 우리는 주어진 벡터공간에 대한 basis vector집합을 구할 수 있다. 예를 들어 가우스 소거법을 통해 해당 공간이 $\operatorname{span}([\tilde{b}_1, \ldots, \tilde{b}_n])$임을 구했다고 해보자. 물론 이 때의 basis vector는는 non-orthogonal, unnormalized형태이다. 따라서 orthonormal basis를 찾기위해 남아있는 작업은 이들을 orthogonal한 형태로 구성하고 normalize해주는 것이다. 이 과정은 다음 과정을 통해 구할 수 있다.

1. Basis를 $\tilde{\boldsymbol{B}} = [\tilde{\boldsymbol{b}}_1, \ldots, \tilde{\boldsymbol{b}}_n]$과 같이 구성하고 augmented matrix를 $[\tilde{\boldsymbol{B}}\tilde{\boldsymbol{B}}^\top \vert \tilde{\boldsymbol{B}}]$로 구성한다.
2. $[\tilde{\boldsymbol{B}}\tilde{\boldsymbol{B}}^\top \vert \tilde{\boldsymbol{B}}]$에 대해 가우스 소거법을 적용한다.
3. Row echolon form을 통해 orthogonal basis를 구한다. Orthogonal basis는 row방향으로 구해짐에 유의한다.
4. Normalize해 orthonormal basis로 바꾸어준다.

위의 과정은 이후 다룰 Gram-Schmidt Process를 가우스 소거법으로 풀어낸 것과 같다. 다음 예제를 통해 확인해보자.

### Example

Basis가 $\boldsymbol{b}_1 = [2, 1]^\top, \boldsymbol{b}_2 = [1, 2]^\top$로 주어졌을 때 orthonormal basis를 만들어보자.

$$
[\tilde{\boldsymbol{B}}\tilde{\boldsymbol{B}}^\top \vert \tilde{\boldsymbol{B}}] = \left[{\begin{array}{rr|rr}5&4&2&1\\
4&5&1&2\end{array}}\right]
$$

이를 Row echolon form으로 변환하면,

$$
\left[{\begin{array}{rr|rr}5&4&2&1\\
0&1&-\frac{1}{3}&\frac{2}{3}\end{array}}\right]
$$

따라서 Orthonormal vector는 다음과 같이 계산할 수 있다.

$$
\boldsymbol{b}_1 = \frac{1}{\sqrt{2^2 + 1^2}}[2, 1]^\top = \frac{1}{\sqrt{5}}[2, 1]^\top
$$

$$
\boldsymbol{b}_2 = \frac{1}{\sqrt{\left(-\frac{1}{3}\right)^2 + \left(\frac{2}{3}\right)^2}}\left[-\frac{1}{3}, \frac{2}{3}\right]^\top = \frac{1}{\sqrt{5}}[-1, 2]^\top
$$


## Why orthogonal basis is so special?

직교성에 대해 앞으로도 수 많은 언급이 있지만, 보다 원론적인 질문을 해보자. 왜 직교, 즉 내적이 0인 basis는 다른 basis 구성보다 더 특별할까?

다음의 예시를 통해 알아보자. 3차원 공간이 있고 서로 독립인 basis가 $\{\boldsymbol{v}_1, \boldsymbol{v}_2, \boldsymbol{v}_3\}$라고 해보자. 이 때, 해당 벡터공간의 임의의 벡터 $\boldsymbol{x}$는 다음과 같이 **유일하게** 표현될 수 있다.

$$
\boldsymbol{x} = \alpha_1 \boldsymbol{v}_1 + \alpha_2 \boldsymbol{v}_2 + \alpha_3 \boldsymbol{v}_3, \quad \alpha_1, \alpha_2, \alpha_3 \in \mathbb{R}
$$

이 식에서 $\alpha_1, \alpha_2, \alpha_3$는 좌표에 대응되는 개념이다. 그렇다면 좌표는 어떻게 계산해야 할까? 만약 $\{\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3\}$가 orthonormal한 basis라면 projection의 성질을 통해 다음의 연산으로 정확한 좌표를 구할 수 있게 된다.

$$
\boldsymbol{x} = (\boldsymbol{v} \cdot \boldsymbol{e}_1)\boldsymbol{e}_1 + (\boldsymbol{v} \cdot \boldsymbol{e}_2)\boldsymbol{e}_2 + (\boldsymbol{v} \cdot \boldsymbol{e}_3)\boldsymbol{e}_3
$$

이처럼 orthogonal할 경우 간단한 연산만으로 좌표를 구할 수 있다. Orthogonal하지 않았다면 훨씬 복잡하게 표현되었을 것이다. 이처럼 orthogonal basis로 공간을 표현하면 분석과 표현에 있어 용이한 점이 많다.

## Conclusion

이 문서에서는 orthogonal basis가 normalize된 형태인 orthonormal basis의 정의를 알아보았고 행렬에서 orthonormal basis를 구하는 법에 대해 다루었다.

## Reference

* Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge, United Kingdom: Cambridge University Press.
