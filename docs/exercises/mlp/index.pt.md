---
exercise: mlp
ai_use: "Usei o Claude para escrever uma primeira versão dos scripts e deste texto, e para revisar a minha análise. Depois reescrevi um pouco do código, rodei todos os scripts e conferi cada parâmetro, fórmula e número contra o enunciado. Eu entendo e sei explicar cada parte do que está sendo entregue."
---

# Exercício 3: MLP

> Entrega: 25/10/2026 (domingo), commits até 23:59.

## Abordagem e reprodutibilidade

O Exercício 1 é feito na mão, nesta página. Não tem script para ele.

Os Exercícios 2, 3 e 4 compartilham o [`base.py`](https://github.com/kikepuppi/neuralnetworks-and-deeplearning-insper/blob/main/docs/exercises/mlp/code/base.py) com a semente, os caminhos e as figuras, e o [`mlp.py`](https://github.com/kikepuppi/neuralnetworks-and-deeplearning-insper/blob/main/docs/exercises/mlp/code/mlp.py) com a rede. Cada script roda sozinho a partir de um clone limpo:

```bash
python -m pip install -r requirements.txt
python docs/exercises/mlp/code/ex2_blobs.py
python docs/exercises/mlp/code/ex3_multiclass.py
python docs/exercises/mlp/code/ex4_depth.py
```

Cada script começa o estado aleatório com `np.random.default_rng(42)`, então os números saem idênticos em toda execução. O Exercício 4 é a única exceção, porque o item B pede três sementes.

O `mlp.py` é usado pelos três exercícios sem mudanças. O forward, o backward e o update são o mesmo código. O que muda entre eles é a lista de argumentos: os tamanhos das camadas, a ativação de saída e a perda. O `scikit-learn` aparece só para gerar os dados, dividir, fazer a matriz de confusão e o baseline linear.

A parte que deu trabalho de verdade foi fazer um backward só servir para duas perdas diferentes. Só funciona porque as duas terminam em `p - y`, que é o que o item 3B deriva.

## Exercício 1
Um passo de backpropagation, na mão.

Com `x = [0.5, -0.2]`, `y = 1.0`, `W¹ = [[0.3, -0.1], [0.2, 0.4]]`, `b¹ = [0.1, -0.2]`, `W² = [0.5, -0.3]`, `b² = 0.2` e `η = 0.3`.

### A: Forward

`z¹ = W¹ x + b¹`

```
z¹₁ = (0.3)(0.5) + (-0.1)(-0.2) + 0.1 = 0.15 + 0.02 + 0.10 = 0.2700
z¹₂ = (0.2)(0.5) + ( 0.4)(-0.2) - 0.2 = 0.10 - 0.08 - 0.20 = -0.1800
```

`h¹ = tanh(z¹)`

```
h¹₁ = tanh( 0.2700) =  0.2636
h¹₂ = tanh(-0.1800) = -0.1781
```

`u² = W² h¹ + b²`

```
u² = (0.5)(0.2636) + (-0.3)(-0.1781) + 0.2 = 0.1318 + 0.0534 + 0.2000 = 0.3852
```

`ŷ = tanh(u²)` e `L = (y - ŷ)²`

```
ŷ = tanh(0.3852) = 0.3672
L = (1.0 - 0.3672)² = (0.6328)² = 0.4004
```

### B: Backward

Partindo da perda e encadeando para fora.

`∂L/∂ŷ = -2(y - ŷ)`

```
∂L/∂ŷ = -2(1.0 - 0.3672) = -1.2655
```

`∂L/∂u² = (∂L/∂ŷ)(1 - ŷ²)`, usando `d tanh(u)/du = 1 - tanh²(u)`

```
1 - ŷ² = 1 - (0.3672)² = 1 - 0.1348 = 0.8652
∂L/∂u² = (-1.2655)(0.8652) = -1.0948
```

Camada de saída. Como `u² = W² h¹ + b²`, então `∂u²/∂W² = h¹` e `∂u²/∂b² = 1`:

```
∂L/∂W² = (∂L/∂u²) h¹ = (-1.0948)[0.2636, -0.1781] = [-0.2886, 0.1950]
∂L/∂b² = ∂L/∂u² = -1.0948
```

Voltando uma camada. Como `∂u²/∂h¹ = W²`:

```
∂L/∂h¹ = (∂L/∂u²) W² = (-1.0948)[0.5, -0.3] = [-0.5474, 0.3285]
```

`∂L/∂z¹ = (∂L/∂h¹)(1 - (h¹)²)`, elemento a elemento:

```
1 - (h¹₁)² = 1 - 0.0695 = 0.9305    ∂L/∂z¹₁ = (-0.5474)(0.9305) = -0.5094
1 - (h¹₂)² = 1 - 0.0317 = 0.9683    ∂L/∂z¹₂ = ( 0.3285)(0.9683) =  0.3180
```

Camada escondida. Como `z¹ = W¹ x + b¹`, cada linha de `∂L/∂W¹` é `∂L/∂z¹ᵢ` vezes `x`:

```
∂L/∂W¹ = [[(-0.5094)(0.5), (-0.5094)(-0.2)],     [[-0.2547,  0.1019],
          [( 0.3180)(0.5), ( 0.3180)(-0.2)]]  =   [ 0.1590, -0.0636]]

∂L/∂b¹ = ∂L/∂z¹ = [-0.5094, 0.3180]
```

### C: Atualização dos parâmetros

Cada parâmetro anda `-η` vezes o seu gradiente, com `η = 0.3`.

```
W¹₁₁ = 0.3 - (0.3)(-0.2547) =  0.3764      W¹₁₂ = -0.1 - (0.3)( 0.1019) = -0.1306
W¹₂₁ = 0.2 - (0.3)( 0.1590) =  0.1523      W¹₂₂ =  0.4 - (0.3)(-0.0636) =  0.4191

b¹₁  = 0.1 - (0.3)(-0.5094) =  0.2528      b¹₂  = -0.2 - (0.3)( 0.3180) = -0.2954

W²₁  = 0.5 - (0.3)(-0.2886) =  0.5866      W²₂  = -0.3 - (0.3)( 0.1950) = -0.3585

b²   = 0.2 - (0.3)(-1.0948) =  0.5284
```

#### Diga se a perda sobe ou desce numa segunda passada, e por quê

Desce. A descida de gradiente move cada parâmetro contra o seu gradiente, então para um `η` pequeno o bastante a perda tem que cair.

A segunda passada confirma:

```
z¹ = [ (0.3764)(0.5) + (-0.1306)(-0.2) + 0.2528,
       (0.1523)(0.5) + ( 0.4191)(-0.2) - 0.2954 ] = [0.4671, -0.3030]

h¹ = [0.4359, -0.2941]

u² = (0.5866)(0.4359) + (-0.3585)(-0.2941) + 0.5284 = 0.8895

ŷ  = tanh(0.8895) = 0.7112

L  = (1.0 - 0.7112)² = 0.0834
```

A perda cai de **0.4004** para **0.0834**, e o `ŷ` sai de 0.3672 na direção do alvo 1.0.

## Exercício 2
Classificação binária que uma reta não resolve.

### A: Gerar os dados

`make_blobs` com os três centros do enunciado, 500 pontos no aglomerado do meio e 250 em cada um dos de fora. A classe 1 são os dois de fora. Divisão 80/20, estratificada, semente 42.

| Divisão | Amostras | Fração da classe 1 |
|---|---|---|
| Treino | 800 | 0.5000 |
| Teste | 200 | 0.5000 |

![Figura 1](figures/fig1.png)

### B: Estabelecer o baseline

A regressão logística treinada no treino chega a **0.4750** no teste, pior que chutar.

### C: Implementar a MLP

Uma camada escondida de 8 unidades com `tanh`, uma unidade de saída com sigmoid, entropia cruzada binária, descida de gradiente no lote inteiro.

| Configuração | Valor |
|---|---|
| Arquitetura | **[2, 8, 1]** |
| Parâmetros | 33 |
| Taxa de aprendizado | 0.5 |
| Épocas | 2000 |
| Perda de treino | 0.7930 para **0.1043** |
| Acurácia no teste | **0.9150** |

![Figura 2](figures/fig2.png)

![Figura 3](figures/fig3.png)

### D: Análise

#### Por que o baseline linear fica abaixo de 50% enquanto a sua MLP passa de 90%?

Porque os dois aglomerados da classe 1 ficam em lados opostos da classe 0, então puxam os pesos da regressão logística em direções opostas e as duas puxadas se cancelam. O vetor ajustado colapsa para quase nada, `‖w‖ = 0.0297` com intercepto -0.0031, o que é um modelo quase plano e responde perto de uma moeda. No teste ele previu classe 0 para 111 pontos e classe 1 para 89.

Um modelo quase plano cai perto de 50% sem mirar nisso, e 0.4750 é o lado em que ele calhou de cair.

#### Descreva o formato da região que a sua rede atribui à classe 1 e explique por que nenhuma reta sozinha produz esse formato

Na Figura 3 a classe 1 recebe duas regiões separadas, o canto superior direito e o canto inferior esquerdo, com a faixa da classe 0 cortando na diagonal entre as duas. A região da classe 1 é desconexa.

Uma reta corta o plano em dois semiplanos, e um semiplano é uma peça conexa só. Então uma reta nunca consegue dar à classe 1 dois cantos separados mantendo a classe 0 no meio. A rede desenha a faixa porque cada unidade escondida contribui com a sua própria reta e a camada de saída combina elas, então dá para manter a classe 0 entre duas delas.

### Código

```python
--8<-- "docs/exercises/mlp/code/base.py"
```

```python
--8<-- "docs/exercises/mlp/code/mlp.py"
```

```python
--8<-- "docs/exercises/mlp/code/ex2_blobs.py"
```

## Exercício 3
Multiclasse, mesma rede.

### A: Gerar os dados

`make_classification` com 1500 amostras, 4 features informativas, 3 classes e 2 aglomerados por classe. Divisão 80/20, estratificada, semente 42.

Isso dá 1200 amostras de treino e 300 de teste, com frações de classe no treino de 0.3325, 0.3325 e 0.3350.

### B: Adaptar a rede

A camada de saída vai para 3 unidades, o softmax transforma isso em probabilidades, e a entropia cruzada categórica substitui a binária. Nada mais muda.

#### Derive `∂L/∂u` para o par softmax e entropia cruzada

Para uma amostra, com `p = softmax(u)` e `y` em one-hot:

```
p_k = e^(u_k) / Σ_j e^(u_j)        L = -Σ_k y_k log p_k
```

A derivada do softmax é

```
∂p_k/∂u_i = p_k (δ_ki - p_i)
```

Encadeando na perda:

```
∂L/∂u_i = -Σ_k (y_k / p_k)(∂p_k/∂u_i)
        = -Σ_k (y_k / p_k) p_k (δ_ki - p_i)
        = -Σ_k y_k (δ_ki - p_i)
        = -(y_i - p_i Σ_k y_k)
```

O `y` é one-hot, então `Σ_k y_k = 1` e tudo colapsa para

```
∂L/∂u_i = p_i - y_i
```

O `p_k` que vem da derivada do softmax cancela o `1/p_k` que vem da derivada do log, e é isso que deixa o resultado tão simples. Sigmoid com entropia binária também dá `∂L/∂u = p - y`, então no lote os dois viram `(p - y)/N`. Essa é a primeira linha do `backward` no `mlp.py`, e é por isso que o mesmo código serve para os Exercícios 2 e 3.

### C: Treinar e avaliar

| Configuração | Valor |
|---|---|
| Arquitetura | **[4, 16, 3]** |
| Parâmetros | 131 |
| Taxa de aprendizado | 0.5 |
| Épocas | 2000 |
| Perda de treino | 1.1166 para **0.2597** |
| Acurácia no teste | **0.8567** |

![Figura 4](figures/fig4.png)

![Figura 5](figures/fig5.png)

### D: Análise

#### Qual par de classes a rede mais confunde, segundo a Figura 5?

As classes 0 e 1, com **18** erros contando as duas direções.

| Par | Erros nas duas direções |
|---|---|
| 0 e 1 | **18** |
| 1 e 2 | 16 |
| 0 e 2 | 9 |

A classe 1 também é a mais fraca sozinha: 77 dos seus 100 pontos de teste saem certos, contra 90 de cada uma das outras duas.

#### Os dados têm 4 dimensões informativas e 6 aglomerados no total. Dê uma explicação para a confusão que se apoie nessa estrutura

Cada classe são dois aglomerados, então as 3 classes são 6 bolhas espalhadas em 4 dimensões com `class_sep=1.0`, que deixa elas próximas. Uma classe cobre duas regiões que podem ficar longe uma da outra.

Isso faz a vizinhança pesar mais que a classe. Se uma das bolhas da classe 1 calha de cair ao lado de uma bolha da classe 0, todo ponto na sobreposição se perde, e a rede não conserta isso mexendo na fronteira da classe 1 sem abrir mão da outra bolha dela. A classe 1 perde pontos para a 0 e para a 2 em quantidades parecidas, 12 e 11.

#### Reporte o baseline de regressão logística nesses dados e diga o que a camada escondida acrescentou

O baseline fica em **0.6633**, contra **0.8567** da rede, então a camada escondida acrescentou **19.3 pontos**.

A regressão logística traça uma fronteira por classe e por isso dá a cada classe uma região conexa só. Com dois aglomerados por classe isso não tem como dar certo. A camada escondida monta features intermediárias primeiro, o que deixa a camada de saída juntar duas bolhas separadas sob o mesmo rótulo.

### Código

```python
--8<-- "docs/exercises/mlp/code/ex3_multiclass.py"
```

## Exercício 4
Uma segunda camada escondida ajuda?

### A: Treinar a rede mais funda

Mesmos dados, mesma semente, mesma taxa de aprendizado e mesmo número de épocas do Exercício 3. A única mudança é uma camada escondida a mais.

| Configuração | Valor |
|---|---|
| Arquitetura | **[4, 16, 16, 3]** |
| Parâmetros | 403 |
| Taxa de aprendizado | 0.5 |
| Épocas | 2000 |
| Perda de treino | **0.1680** |
| Acurácia no teste | **0.8533** |

### B: Comparar

![Figura 6](figures/fig6.png)

Os dois modelos treinados com as sementes 42, 43 e 44, com todo o resto fixo.

| | Camadas escondidas | Parâmetros | Perda final de treino | Acurácia no teste |
|---|---|---|---|---|
| Exercício 3 | 1 | 131 | 0.2617 | 0.8489 |
| Exercício 4 | 2 | 403 | **0.1723** | **0.8522** |

As duas colunas são a média das três sementes. Por semente as acurácias foram 0.8567, 0.8367 e 0.8533 com uma camada, e 0.8533, 0.8467 e 0.8567 com duas.

### C: Análise

#### Explique o resultado

A rede mais funda ganha na perda de treino e empata na acurácia de teste. Ela corta a perda em **0.0894**, uma queda de 34%, e ganha **0.0033** de acurácia, que é 1 ponto em 300.

Esse ganho é menor que a variação entre sementes. Com uma camada a acurácia variou 0.0200 entre as três sementes, seis vezes a diferença entre os dois modelos. Ou seja, a comparação diz que os dois modelos são o mesmo, e rodar uma semente só poderia ter mostrado qualquer um dos dois ganhando.

Os parâmetros três vezes mais numerosos foram para ajustar o treino, não para generalizar. Na Figura 6 as duas curvas de perda se separam com clareza enquanto a acurácia não sai do lugar.

Os dados explicam o porquê. 6 bolhas gaussianas em 4 dimensões precisam de uma fronteira curva e em pedaços, mas não funda. Uma camada escondida de 16 unidades já entrega isso, e uma segunda camada acrescenta profundidade que o problema não tem como usar, além de deixar a otimização mais difícil.

#### Diga que tipo de dado faria a profundidade extra valer a pena

Dado em que as features úteis são construídas a partir de outras features, em etapas. Imagem é o caso clássico: bordas, depois cantos, depois partes, depois objetos. Cada camada reaproveita o que a de baixo achou, então a profundidade multiplica o que dá para expressar em vez de só alargar.

O formato comum é entrada crua e de alta dimensão com estrutura composicional, como pixels, áudio ou texto. Aqui a entrada são 4 features informativas prontas, sem hierarquia por baixo, então não tem o que uma segunda camada compor.

### Código

```python
--8<-- "docs/exercises/mlp/code/ex4_depth.py"
```

## Resumo dos resultados

| # | Grandeza | Valor |
|---|---|---|
| 1 | Ex. 1, `z¹` e `h¹` | z¹ = [0.2700, -0.1800], h¹ = [0.2636, -0.1781] |
| 2 | Ex. 1, `u²`, `ŷ` e `L` | u² = 0.3852, ŷ = 0.3672, L = 0.4004 |
| 3 | Ex. 1, os oito parâmetros atualizados | W¹ = [[0.3764, -0.1306], [0.1523, 0.4191]], b¹ = [0.2528, -0.2954], W² = [0.5866, -0.3585], b² = 0.5284 |
| 4 | Ex. 2, acurácia do baseline linear no teste | 0.4750 |
| 5 | Ex. 2, arquitetura e acurácia no teste | [2, 8, 1], 0.9150 |
| 6 | Ex. 3, arquitetura e acurácia no teste | [4, 16, 3], 0.8567 |
| 7 | Ex. 3, par de classes mais confundido | classes 0 e 1, 18 erros |
| 8 | Ex. 4, arquitetura e acurácia no teste | [4, 16, 16, 3], 0.8533 |
| 9 | Ex. 4, diferença de acurácia contra o Ex. 3 (média de 3 sementes) | +0.0033 (0.8522 contra 0.8489) |
