# Neural Number Lab

A browser-based neural network that learns to recognize handwritten digits (1–9) from scratch — no server, no pre-trained model, no external ML libraries. Draw a number, watch it become a 28×28 image, and see the network learn in real time.

> **Author:** Luca Filippi

---

## What is it?

Neural Number Lab is an interactive web page where you teach a neural network to recognize handwritten digits by drawing them yourself. Everything runs in your browser — the network is built, trained, and tested entirely on your machine.

## What is it for?

It's a hands-on learning tool for understanding how neural networks actually work: forward propagation, backpropagation, gradient descent, overfitting, and generalization — all made visible through live charts and a network diagram.

## The main idea

You draw, the drawing becomes 784 numbers, the network predicts a digit, it compares with the correct answer, and it adjusts its weights to make fewer mistakes next time. **The network only knows what you taught it.**

---

## Demo



> Video / screenshots coming soon — link will be added here.

---

## Main features

- Draw digits on a canvas with mouse or touch, from 1 to 9
- "What the AI sees" preview showing the 28×28 preprocessed image
- Live training with loss and accuracy charts per epoch
- Adjustable hyperparameters: learning rate, epochs, batch size
- Network visualization showing weights and which neuron fired
- Test tab to check generalization on new drawings
- Teach the AI directly when it gets a prediction wrong
- Save/load: samples, weights and history persist in `localStorage`
- Export/import the whole model as JSON
- Light/dark theme support

---

## How the recognition pipeline works

1. Draw on a 280×280 canvas.
2. Preprocess: convert to grayscale ink, crop to bounding box, scale to 20 px, center by center of mass, normalize, resulting in 784 values (0–1).
3. Forward pass: `784 → 64 → 32 → 9` with ReLU hidden layers and softmax output, producing 9 probabilities.
4. Train: cross-entropy loss + backpropagation + gradient descent in mini-batches.
5. Predict: the highest probability wins.

**Architecture:** 784 → 64 → 32 → 9 · ~52,617 parameters.

---

## Technologies

- HTML5 + CSS3 (custom properties, responsive grid, `prefers-color-scheme`)
- Vanilla JavaScript (no frameworks, no bundlers)
- Canvas API for drawing, previews and network visualization
- LocalStorage + FileReader + Blob for persistence and export/import
- Neural network implemented from scratch in plain JS (`Float32Array` weights)

---

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/LucaFilippi/Neural-Number-Lab
   ```
2. Open `index.html` in any modern browser.

That's it. No build step, no dependencies, no server required.

---

## Requirements

- A modern browser with support for Canvas, Pointer Events, and ES2020+ (Chrome, Firefox, Edge, Safari).

---

## Known limitations

- Few examples lead to poor results. The network memorizes what it sees.
- Variety matters more than quantity. Similar drawings lead to narrow recognition.
- Training accuracy is not real accuracy. Always validate on the TEST tab with new drawings.
- Unbalanced data skews predictions (e.g., many 1s and few 7s leads to a bias toward 1).
- Works best on a single style of handwriting per digit.

---

## Project goal

This project was built as a learning exercise to:

- Understand how an AI is built end-to-end, from raw pixels to predictions.
- Train the ability to write a neural network from scratch, with no libraries hiding the math.
- Experiment with pattern recognition and see the impact of data quality and hyperparameters.

It's meant to be read, broken, and rewritten. The source is organized to mirror the concepts.

---

---

# Neural Number Lab (Português)

Uma rede neural que roda no navegador e aprende a reconhecer números escritos à mão (1 a 9) do zero — sem servidor, sem modelo pré-treinado e sem bibliotecas de ML. Você desenha, vê o desenho virar uma imagem 28×28 e acompanha a rede aprendendo em tempo real.

> **Autor:** Luca Filippi

---

## O que é?

Neural Number Lab é uma página web interativa onde você ensina uma rede neural a reconhecer dígitos desenhando você mesmo. Tudo roda no navegador — a rede é construída, treinada e testada na sua máquina.

## Para que serve?

É uma ferramenta prática de aprendizado para entender como uma rede neural funciona de verdade: propagação direta, backpropagation, descida do gradiente, overfitting e generalização — tudo visível em gráficos ao vivo e no diagrama da rede.

## Qual é a ideia principal?

Você desenha, o desenho vira 784 números, a rede prevê um dígito, compara com a resposta certa e ajusta os pesos para errar menos na próxima. **A rede só sabe o que você ensinou.**

---

## Demonstração


> Vídeo / prints em breve — o link será colocado aqui.

---

## Principais funcionalidades

- Desenhar dígitos no canvas com mouse ou toque (1–9)
- Prévia "O que a IA vê" mostrando a imagem 28×28
- Treino ao vivo com gráficos de perda e acurácia por época
- Hiperparâmetros ajustáveis: taxa de aprendizado, épocas, tamanho do lote
- Visualização da rede com pesos e neurônio vencedor
- Aba TEST para verificar generalização em desenhos novos
- Ensinar a IA na hora quando ela erra
- Salvamento automático no `localStorage` (amostras, pesos, histórico)
- Exportar/importar o modelo inteiro em JSON
- Suporte a tema claro e escuro

---

## Como funciona o reconhecimento

1. Desenho no canvas 280×280.
2. Pré-processamento: escala de cinza, corte pela caixa delimitadora, escala para 20 px, centralização por centro de massa, normalização, resultando em 784 valores (0–1).
3. Forward: `784 → 64 → 32 → 9` com ReLU nas camadas ocultas e softmax na saída, gerando 9 probabilidades.
4. Treino: entropia cruzada + backpropagation + descida do gradiente em mini-lotes.
5. Previsão: a maior probabilidade vence.

**Arquitetura:** 784 → 64 → 32 → 9 · ~52.617 parâmetros.

---

## Tecnologias utilizadas

- HTML5 + CSS3 (variáveis CSS, grid responsivo, `prefers-color-scheme`)
- JavaScript puro (sem frameworks e sem bundlers)
- Canvas API para desenho, prévias e visualização da rede
- LocalStorage + FileReader + Blob para persistência e exportação/importação
- Rede neural implementada do zero em JS puro (pesos em `Float32Array`)

---

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/LucaFilippi/Neural-Number-Lab
   ```
2. Abra o `index.html` em qualquer navegador moderno.

Pronto. Sem build, sem dependências, sem servidor.

---

## Requisitos

- Navegador moderno com suporte a Canvas, Pointer Events e ES2020+ (Chrome, Firefox, Edge, Safari).

---

## Limitações conhecidas

- Poucos exemplos geram resultados ruins. A rede decora o que viu.
- Variedade importa mais que quantidade. Desenhos parecidos levam a um reconhecimento limitado.
- Acurácia de treino não é acurácia real. Valide sempre na aba TEST com desenhos novos.
- Dados desequilibrados distorcem as previsões (ex.: muitos 1s e poucos 7s geram viés para o 1).
- Funciona melhor com um estilo consistente de escrita por dígito.

---

## Objetivo do projeto

Este projeto foi construído como exercício de aprendizado para:

- Entender como uma IA é feita de ponta a ponta, do pixel à previsão.
- Treinar a habilidade de escrever uma rede neural do zero, sem bibliotecas escondendo a matemática.
- Experimentar reconhecimento de padrões e ver o impacto da qualidade dos dados e dos hiperparâmetros.

A ideia é ser lido, quebrado e reescrito. O código está organizado para espelhar os conceitos.

---

<p align="center">
  Made by <b>Luca Filippi</b>
</p>