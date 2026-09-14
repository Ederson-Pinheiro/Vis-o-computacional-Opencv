# Visao-computacional-Opencv
Projeto do segundo módulo do SCTEC. Pré-processamento de imagens com OpenCV.

# Mini-Projeto — Visão Computacional

## Objetivo

Desenvolver um pipeline de pré-processamento de imagens utilizando **Python e OpenCV**, preparando imagens de peças para uma futura aplicação de Machine Learning.

O projeto não realiza classificação. O objetivo é somente realizar o pré-processamento das imagens.

## Tecnologias

* Python
* OpenCV
* NumPy
* TQDM
* Git e GitHub

## Pipeline

O processamento das imagens segue as seguintes etapas:

```text
Imagem original
      ↓
Grayscale
      ↓
Gaussian Blur
      ↓
Threshold Otsu
      ↓
Morfologia
      ↓
Canny
      ↓
Resize 256x256
      ↓
Salvamento
```

### Grayscale

Converte a imagem colorida para escala de cinza.

### Gaussian Blur

Reduz ruídos da imagem antes das etapas de segmentação.

### Threshold Otsu

Realiza a segmentação da imagem utilizando um limiar automático.

### Morfologia

Utiliza `MORPH_OPEN` para auxiliar na remoção de pequenos ruídos.

### Canny

Realiza a detecção das bordas presentes na imagem.

### Resize

Todas as imagens são redimensionadas para **256 × 256 pixels**.

### Salvamento

As imagens processadas são salvas na pasta:

```text
data/processed_images/
```

## Orientação a Objetos

O projeto utiliza a classe `ImageProcessor` para organizar o pipeline.

Principais métodos:

```text
preprocess()
segment()
morphology()
resize()
process_image()
process_batch()
```

A execução é realizada através de:

```python
processor = ImageProcessor(
    "data/raw_images",
    "data/processed_images"
)

processor.process_batch()
```

## Instalação

Criar o ambiente virtual:

```bash
python -m venv .venv
```

Ativar no Windows:

```bash
.venv\Scripts\activate
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

## Como executar o projeto

Após instalar as dependências, coloque as imagens originais na pasta:

```text
data/raw_images/
```

Depois, execute o arquivo principal pelo terminal, a partir da pasta raiz do projeto:

```bash
python src/main.py
```

O programa irá:

1. Ler as imagens da pasta `data/raw_images/`;
2. Aplicar o pipeline de pré-processamento;
3. Redimensionar as imagens para `256x256`;
4. Salvar os resultados em `data/processed_images/`.

Após a execução, as imagens processadas estarão disponíveis em:

```text
data/processed_images/
```

## Git

O projeto foi desenvolvido utilizando Git e organizado em branch de desenvolvimento.

Exemplos de commits:

```text
comeco: configura ambiente inicial
feat: implementa leitura em lote
feat: adiciona grayscale e gaussian blur
feat: adiciona threshold otsu
feat: adiciona operacoes morfologicas
feat: adiciona deteccao de bordas
feat: adiciona resize e salvamento
docs: adiciona README
```

## Possíveis melhorias

Como próximos passos, o projeto pode receber:

* melhoria da segmentação;
* testes com outros filtros;
* ajustes nos parâmetros do Canny;
* identificação e separação das peças;
* extração de características;
* treinamento de um modelo de Machine Learning.

## Conclusão

O projeto implementa um pipeline de pré-processamento de imagens utilizando **Python e OpenCV**, gerando imagens padronizadas em **256 × 256 pixels** para utilização em futuras etapas de Machine Learning.
