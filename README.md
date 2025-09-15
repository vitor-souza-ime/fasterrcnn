# Faster R-CNN para Detecção de Objetos em Imagens

Este repositório contém uma implementação em Python do **Faster R-CNN** para detecção de objetos em imagens, utilizando o modelo pré-treinado **FasterRCNN ResNet50 FPN** da biblioteca PyTorch. O projeto permite carregar imagens a partir de URLs, realizar inferência do modelo e visualizar os resultados com **bounding boxes** e **scores de confiança**.

O código foi desenvolvido com foco em **detecção de veículos e pessoas**, mas pode ser adaptado para outras classes de objetos presentes no dataset **COCO**.

Repositório: [https://github.com/vitor-souza-ime/fasterrcnn](https://github.com/vitor-souza-ime/fasterrcnn)

---

## Estrutura do repositório

```

fasterrcnn/
├── main.py          # Código principal para detecção de objetos
├── README.md        # Este arquivo
└── requirements.txt # Dependências do projeto

````

---

## Instalação

1. Clone este repositório:

```bash
git clone https://github.com/vitor-souza-ime/fasterrcnn.git
cd fasterrcnn
````

2. Crie e ative um ambiente virtual (opcional, recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install torch torchvision pillow matplotlib requests
```

> Obs.: Certifique-se de ter **PyTorch** instalado com suporte à sua GPU, se desejar aceleração por CUDA. Consulte [PyTorch Installation](https://pytorch.org/get-started/locally/) para instruções específicas.

---

## Uso

1. Abra o arquivo `main.py` e altere a variável `image_url` para a imagem que deseja processar.

```python
image_url = "URL_DA_SUA_IMAGEM"
```

2. Execute o script:

```bash
python main.py
```

3. O script irá:

   * Carregar a imagem da URL fornecida
   * Realizar a detecção de objetos usando Faster R-CNN
   * Desenhar **bounding boxes** e **probabilidades** de detecção
   * Exibir a imagem com os resultados

---

## Classes Detectadas

O modelo utiliza **COCO** como base de treinamento, portanto pode detectar as seguintes classes (exemplos):

* pessoa, bicicleta, carro, motocicleta, ônibus, caminhão, avião, gato, cachorro, etc.

---

## Personalização

Você pode ajustar:

* **Threshold de confiança**:

```python
image_with_boxes = draw_boxes(image, prediction, threshold=0.5)
```

* **Modelo**: é possível usar outros modelos pré-treinados disponíveis no `torchvision.models.detection`.

* **Anotação de imagens locais**: modificar `load_image_from_url` para carregar imagens do disco local.

---

## Referência

* Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
* PyTorch Documentation: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
