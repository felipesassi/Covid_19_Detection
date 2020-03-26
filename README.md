# Covid-19 Detection


## O problema proposto

A ideia básica desse projeto é realizar a classificação de células em duas classes, com metástase e saudáveis. O conjunto de dados é composto por images de 96 pixels de largura por 96 pixels de altura.

> In this competition, you must create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans.

## Conjuntos de dados

As imagens que compõe o conjunto de dados são como a imagem abaixo.

![image](images/example.png)

### Conjunto de dados de treino

O conjunto de dados de treino é composto por 220025 imagens.

### Conjunto de dados de teste

O conjunto de testes é composto por 57458 imagens

### Avaliação do modelo

A métrica utilizada para avaliar o modelo é a mesma definida pela competição:

> Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Descrição da solução

Todo o projeto foi desenvolvido por meio da plataforma Google Colab, a qual disponibiliza acesso a uma GPU de alto desempenho gratuitamente.

Diversos modelos de redes neurais convolucioonais foram testados afim de se maximizar o desempenho obtido na classificação. O modelo escolhido consiste um um rede ResNet50 com a inserção de três camadas completamente conectadas para a classificação. Além disso, os pesos da parte convolucional são inicializados com os pesos utilizados na competição ImageNet.

Para melhorar o desempenho do modelo algumas técnicas de *data agumentation* foram utilizadas (os nomes correspondem as funções da biblioteca *albumentations*):

- RGBShift();
- Blur();
- RandomGamma();
- RandomBrightness();
- RandomContrast();
- VerticalFlip();
- HorizontalFlip();
- Normalize();
- CoarseDropout().

Quatro experimentos foram realizados:

- Experimento 1: sem *data augmentation*;
- Experimento 2: com *data augmentation* (RandomGamma(), RandomBrightness(), RandomContrast());
- Experimento 3: com *data augmentation* (VerticalFlip(), HorizontalFlip());
- Experimento 4: com todas as técnicas de *data augmenation* mencionadas.

Em todos os experimentos as imagens de entrada foram normalizadas.

A função custo utilizada é BCEWithLogitsLoss() a qual incorpora uma camada sigmóide e a função custo BCELoss().	

### ResNet50 modificada

A arquitetura ResNet50 utilizada nesse projeto foi a seguinte:

![image](images/model.png)

## Resultados

O modelo conseguiu os seguintes resultados em cada um dos experimentos:

| Experiment | Train | Validation | Test |
|------------|-------|------------|------|
| 1          | 0.98  | 0.98       | 0.95 |
| 2          | 0.98  | 0.99       | 0.95 |
| 3          | 0.98  | 0.98       | 0.94 |
| 4          | 0.96  | 0.98       | 0.94 |

###

## Como usar

### Requirements

- Python 3;
- Pytorch;
- NumPy;
- Albumentations;
- SciKit Learn;
- Pandas.

### Dowload de dados

Após clonar o repositório é necessária realizar o donwload do conjunto de dados e descompactar tudo dentro de uma chamada *data*.

### Treinamento do modelo

Para treinar o modelo é necessário rodar o aquivo *train.py* atráves do seguinte comando:

```
python3 train.py
```

### Criação do conjunto de teste

Para gerar o conjunto de dados a ser enviado para o Kaggle é necessário rodar o arquivo *eval.py* por meio do seguinte comando:

```
python3 eval.py
```

## Descrição dos arquivos do projeto

### Datasets

Essa pasta contém todas as funcionalidades necessárias para manipulação dos dados. Aqui são criados os *data loaders* para possibilitar o treinamento e a avaliação em *batches* do modelos.

### Models

Aqui ficam os arquivos resposáveis pela arquitetura do modelo e por todo o controle do treinamento/avaliação.

### Utils

Aqui ficam arquivos de utilidades, como funções para liberar a memória da GPU.

### Metrics

Aqui são definidas as métricas a serem utilizadas no projeto.

### Optimizers

Nessa pasta são definidos otimizados customizados.

### Configs

Nessa pasta fica um arquivo *.json* com as principais configurações para a rede neural e para o treinamento.


