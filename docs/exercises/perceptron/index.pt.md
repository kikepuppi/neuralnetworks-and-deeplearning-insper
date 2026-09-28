---
exercise: perceptron
ai_use: "Usei o Claude para escrever uma primeira versão dos scripts e deste texto, e para revisar a minha análise. Depois reescrevi um pouco do código, rodei todos os scripts e conferi cada parâmetro, fórmula e número contra o enunciado. Eu entendo e sei explicar cada parte do que está sendo entregue."
---

# Exercício 2: Perceptron

> Entrega: 29/09/2026 (terça), commits até 23:59.

## Abordagem e reprodutibilidade

Um script por exercício. Eles compartilham o [`base.py`](https://github.com/kikepuppi/neuralnetworks-and-deeplearning-insper/blob/main/docs/exercises/perceptron/code/base.py) com a semente, os caminhos, a geração dos dados e os gráficos, e o [`perceptron.py`](https://github.com/kikepuppi/neuralnetworks-and-deeplearning-insper/blob/main/docs/exercises/perceptron/code/perceptron.py) com o modelo. Cada script roda sozinho a partir de um clone limpo:

```bash
python -m pip install -r requirements.txt
python docs/exercises/perceptron/code/ex1_separable.py
python docs/exercises/perceptron/code/ex2_overlap.py
```

Cada script começa o estado aleatório com `np.random.default_rng(42)` e usa esse mesmo gerador até o fim, então os números saem idênticos em toda execução. O perceptron é escrito do zero com numpy. Nenhuma biblioteca fornece o modelo.

A mesma classe `Perceptron` é usada nos dois exercícios, com o pocket já dentro dela. O Exercício 1 não precisa do pocket, então só ignora, e o Exercício 2 reusa a classe sem mudar nada.

O que deu mais trabalho não foi o laço de treino e sim ler os resultados. No Exercício 2 uma acurácia de 50% parece um bug, e o jeito de ver que não é conferir onde a fronteira parou, em vez de olhar só o número.

## Exercício 1
Dados separáveis: o caso para o qual o perceptron foi feito.

### A: Gerar os dados

2000 amostras, 1000 por classe, tiradas de normais multivariadas em 2D.

| Classe | Média | Covariância |
|---|---|---|
| 0 | [1.5, 1.5] | [[0.5, 0], [0, 0.5]] |
| 1 | [5, 5] | [[0.5, 0], [0, 0.5]] |

![Figura 1](figures/fig1.png)

### B: Implementar o perceptron

A predição é `ŷ = step(w · x + b)`, com `step(z) = 1` quando `z >= 0` e `0` caso contrário.

O update roda amostra por amostra: `w ← w + η (y - ŷ) x` e `b ← b + η (y - ŷ)`. Com rótulos em `{0, 1}` o erro `y - ŷ` é `0` quando acerta, então só os erros mexem nos pesos.

Os pesos começam em `w = [0.002532, 0.008952]`, tirados de `rng.normal(0, 0.01, size=2)`, e `b = 0`. A taxa de aprendizado é `η = 0.01`. O treino para na primeira passada completa sem nenhum update, ou em 100 épocas.

### C: Treinar e medir

| Grandeza | Valor |
|---|---|
| `w` final | **[0.0505, 0.0289]** |
| `b` final | **-0.2500** |
| Épocas | **26** |
| Acurácia final | **1.0000** (100%) |
| Pontos classificados errado | 0 |

A época 26 é a que não fez nenhum update e parou o laço. A última mudança de peso foi na época 25.

![Figura 2](figures/fig2.png)

A Figura 2 não tem marcação de erro porque não existe nenhum.

![Figura 3](figures/fig3.png)

### D: Análise

#### Por que dados separáveis convergem rápido? Ligue a resposta à regra de update: o que acontece com o número de updates por época conforme o treino avança?

Porque um acerto dá `y - ŷ = 0` e não produz update. Só os erros mexem nos pesos, então o laço é movido pelos próprios erros e se desliga sozinho quando não há mais nenhum.

O número de updates por época já é pequeno no começo e depois zera:

| Época | 1 | 5 | 10 | 15 | 20 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|
| Updates | 3 | 3 | 4 | 3 | 2 | 2 | 3 | 1 | **0** |

Não é uma queda suave. Fica entre 2 e 4 na maior parte do treino, cai para 1 na época 25 e para 0 na 26. As nuvens estão longe uma da outra, então só um punhado de pontos perto da fronteira erra, e cada passada empurra a reta um pouco mais até nenhum errar.

#### Rode o treino de novo com `η = 1.0`, sem mudar mais nada. Reporte o número de épocas e a acurácia final, e compare a direção de `w` com a do treino com `η = 0.01`

Os dois treinos partem do mesmo `w` e do mesmo `b`. Só o `η` muda.

| Treino | `w` | `b` | Épocas | Acurácia | Direção `w / ‖w‖` |
|---|---|---|---|---|---|
| `η = 0.01` | [0.0505, 0.0289] | -0.2500 | **26** | **1.0000** | [0.8681, 0.4963] |
| `η = 1.00` | [5.8706, 3.3592] | -31.0000 | **37** | **1.0000** | [0.8679, 0.4967] |

As duas direções estão a **0.02 grau** uma da outra, ou seja, são a mesma direção na prática. As fronteiras ainda são retas diferentes, porque o deslocamento muda: `b / ‖w‖` é -4.2979 contra -4.5832. E o número de épocas muda, 26 contra 37.

#### O que o `η` controla, sabendo que cada update soma `η x` a pesos que começaram com magnitude de mais ou menos 0.01?

Ele controla o tamanho do passo em relação aos pesos iniciais, que são a única coisa que o `η` não escala.

A `‖x‖` média aqui é 4.6543, então um update soma mais ou menos `0.01 × 4.65 = 0.047` com `η = 0.01` e mais ou menos `4.65` com `η = 1.0`. O `w` inicial tem magnitude 0.009303. Então o começo aleatório é uns 20% de um passo no treino lento e 0.2% de um passo no rápido.

É por isso que os dois treinos levam números diferentes de épocas e terminam apontando para o mesmo lado. Os dados decidem a direção. O `η` decide o quanto o começo aleatório ainda pesa no caminho.

#### Mostre algebricamente que treinar duas vezes, a partir de `w = 0` e `b = 0`, com `η₁` e `η₂`, dá pesos que diferem só pelo fator `η₂ / η₁`

Partindo de zero, depois de qualquer sequência de erros `(x₁, e₁), …, (x_t, e_t)`:

`w = η Σ eᵢ xᵢ` e `b = η Σ eᵢ`

Pegue dois treinos com `η₁` e `η₂`, e suponha que eles cometeram os mesmos erros até agora. Então `w₂ = (η₂/η₁) w₁` e `b₂ = (η₂/η₁) b₁`, e o fator `η₂/η₁` é positivo.

A próxima predição é `step(w · x + b)`, que só olha se `w · x + b >= 0`. Multiplicar por uma constante positiva não muda isso, nem no empate `z = 0`. Então os dois treinos preveem a mesma coisa, cometem o mesmo erro, e a relação continua valendo depois do update. Por indução vale até a última época.

Mesmos erros quer dizer mesmo número de épocas, e `w · x + b = 0` e `(η₂/η₁)(w · x + b) = 0` são a mesma reta. Então o `η` não mudaria nada, que é justamente o motivo do item B pedir um começo diferente de zero.

Rodando, a álgebra se confirma:

| Treino a partir de `w = 0` | `w` | `b` | Épocas |
|---|---|---|---|
| `η = 0.01` | [0.058681, 0.033503] | -0.310000 | 37 |
| `η = 1.00` | [5.868084, 3.350287] | -31.000000 | 37 |
| Razão | [100.0, 100.0] | 100.0 | iguais |

### Código

```python
--8<-- "docs/exercises/perceptron/code/base.py"
```

```python
--8<-- "docs/exercises/perceptron/code/perceptron.py"
```

```python
--8<-- "docs/exercises/perceptron/code/ex1_separable.py"
```

## Exercício 2
Dados sobrepostos: o caso que o perceptron não resolve.

### A: Gerar os dados

As mesmas 1000 amostras por classe, médias bem mais próximas e espalhamento três vezes maior.

| Classe | Média | Covariância |
|---|---|---|
| 0 | [3, 3] | [[1.5, 0], [0, 1.5]] |
| 1 | [4, 4] | [[1.5, 0], [0, 1.5]] |

![Figura 4](figures/fig4.png)

### B: Treinar guardando os melhores pesos

A mesma classe do Exercício 1, mesmo `η = 0.01`, mesmo teto de 100 épocas, mesma inicialização. O pocket guarda uma cópia de `(w, b)` toda vez que um update produz uma acurácia no conjunto inteiro maior que qualquer uma vista antes.

| Pesos | `w` | `b` | Acurácia |
|---|---|---|---|
| Finais | [0.0361, 0.0494] | -0.0400 | **0.5005** |
| Pocket | [0.0068, 0.0066] | -0.0500 | **0.7285** |

O melhor do pocket apareceu na **época 18**, e o treino rodou as 100 épocas inteiras sem nunca parar.

### C: Figuras

![Figura 5](figures/fig5.png)

Os pesos finais erram 999 dos 2000 pontos, os do pocket erram 543.

![Figura 6](figures/fig6.png)

### D: Análise

#### Explique a diferença entre a acurácia do pocket e a dos pesos finais. Onde a fronteira final fica em relação à nuvem de dados, e por que o laço de treino a deixa lá?

A fronteira final fica fora da nuvem, no canto inferior esquerdo da Figura 5. Quase tudo cai de um lado só dela, então o modelo responde 1 para quase todo ponto: acerta os 1000 da classe 1 e erra 999 dos 1000 da classe 0. É isso o 0.5005.

O motivo está no tamanho dos dois passos. Um erro move o `b` em `η = 0.0100`, mas move o `w` em `η ‖x‖ = 0.0507`, umas cinco vezes mais. Para cortar uma nuvem centrada em `[3.5, 3.5]`, o `b` precisa cancelar `w · [3.5, 3.5]`, e ele não acompanha um `w` que cresce cinco vezes mais rápido.

Os números dizem o mesmo:

| Pesos | `w · [3.5, 3.5] + b` |
|---|---|
| Finais | **0.2592** |
| Pocket | **-0.0029** |

O valor do pocket é praticamente zero, então aquela fronteira passa pelo meio dos dados. O valor final é claramente positivo, então o centro da nuvem está do lado positivo e quase tudo em volta também.

O laço deixa a fronteira lá porque não tem motivo para não deixar. Ele não minimiza nada. Ele reage ao último ponto que viu, e as amostras estão guardadas com a classe 0 primeiro e a classe 1 depois, então a última coisa que toda época faz é pegar uma sequência de erros da classe 1 e empurrar a fronteira para longe.

#### Compare a Figura 3 com a Figura 6. O que o teorema de convergência do perceptron garante, e qual das suas hipóteses este conjunto viola?

Na Figura 3 a curva sobe e estabiliza em 1.0, e o laço para sozinho na época 26. Na Figura 6 a curva azul nunca estabiliza. Ela fica entre 0.5005 e 0.5235 nas 100 épocas e só para porque o teto mandou.

O teorema garante que o perceptron comete um número finito de erros e termina, limitado por `(R/γ)²`, onde `R` é o maior `‖x‖` e `γ` é a margem. Ele supõe que os dados são linearmente separáveis com margem `γ` positiva.

Este conjunto não tem essa margem. As nuvens se sobrepõem, então nenhuma reta deixa todos os pontos do lado certo, `γ` não existe e o limite não quer dizer nada.

#### Colocar mais épocas resolve? E um `η` menor?

Nenhum dos dois.

Mais épocas não ajudam porque não existe estado em que o laço para. Sempre tem algum ponto errado, então toda época produz updates e os pesos continuam se mexendo. A Figura 6 mostra a acurácia atual parada em uns 50% da época 20 até a 100, e o pocket não melhora desde a época 18.

Um `η` menor não ajuda porque ele escala `w` e `b` juntos. A razão entre os dois passos continua `1 : ‖x‖` seja qual for o `η`, então o deslocamento fica para trás exatamente igual. É o mesmo argumento do começo em zero do Exercício 1: encolher todo passo pelo mesmo fator leva os pesos para uma escala menor, não para uma reta melhor.

O que ajuda é o pocket, e ele não é uma mudança na regra de update. Ele só para de jogar fora os bons pesos pelos quais o laço passa.

### Código

```python
--8<-- "docs/exercises/perceptron/code/ex2_overlap.py"
```

## Resumo dos resultados

| # | Grandeza | Valor |
|---|---|---|
| 1 | Exercício 1, `w` e `b` finais | w = [0.0505, 0.0289], b = -0.2500 |
| 2 | Exercício 1, épocas até convergir | 26 |
| 3 | Exercício 1, acurácia final | 1.0000 (100%) |
| 4 | Exercício 1, épocas e acurácia final com `η = 1.0` | 37 épocas, 1.0000 (100%) |
| 5 | Exercício 2, `w` e `b` finais | w = [0.0361, 0.0494], b = -0.0400 |
| 6 | Exercício 2, acurácia dos pesos finais | 0.5005 (50.05%) |
| 7 | Exercício 2, acurácia dos pesos do pocket | 0.7285 (72.85%) |
| 8 | Exercício 2, época em que o melhor do pocket aconteceu | 18 |
