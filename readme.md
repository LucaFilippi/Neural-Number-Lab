# Neural Number Lab

A browser-based neural network that learns to recognize handwritten digits (1–9) from scratch — no server, no pre-trained ML model, and no external machine learning libraries.

Draw a number, watch it become a 28×28 image, and see the neural network learn in real time.

> **Author:** Luca Filippi

---

## What is it?

**Neural Number Lab** is an interactive web application where you teach a neural network to recognize handwritten digits by drawing them yourself.

Everything runs locally in your browser. The network is built, trained, tested, and stored on your machine.

You can either:

* **Train a new model from scratch** using your own drawings.
* **Load one of the pre-trained models** included in the `models/` folder and start testing immediately.

> **The network only knows what it has been taught.**

---

## What is it for?

Neural Number Lab is a hands-on learning project designed to make the fundamentals of neural networks visible and interactive.

It demonstrates concepts such as:

* Forward propagation
* Backpropagation
* Gradient descent
* Cross-entropy loss
* Mini-batch training
* Overfitting
* Generalization
* Training data quality
* Hyperparameter tuning

Instead of hiding the mathematics behind an ML framework, the neural network is implemented from scratch in JavaScript.

---

## The main idea

The process is simple:

**You draw → the drawing becomes 784 values → the network makes a prediction → the prediction is compared with the correct answer → the weights are updated → the network improves.**

The original drawing is a 280×280 canvas, which is processed into a normalized **28×28 image**, resulting in **784 input values**.

The network then processes those values and produces probabilities for the nine possible digits.

---

## Demo

> Video and screenshots coming soon.

---

## Main features

* Draw digits from **1 to 9** using mouse or touch
* "What the AI sees" preview showing the preprocessed 28×28 image
* Real-time training with loss and accuracy charts
* Adjustable hyperparameters:

  * Learning rate
  * Number of epochs
  * Batch size
* Neural network visualization
* Weight visualization
* Visualization of the predicted neuron
* Test tab for evaluating generalization on new drawings
* Teach the AI directly when it makes an incorrect prediction
* Automatic persistence using `localStorage`
* Save samples, weights, and training history
* Export and import the complete model as JSON
* Included pre-trained models
* Light and dark theme support
* Runs entirely in the browser

---

## Pre-trained models

The repository includes a `models/` folder containing models that have already been trained using Neural Number Lab.

This allows you to experiment with the application immediately without having to manually draw and train hundreds of examples first.

### Using a pre-trained model

1. Open the application.
2. Use the model import/load functionality.
3. Select a model from the `models/` folder.
4. Load it into the application.
5. Go to the **TEST** tab and start drawing.

You can also continue training an imported model with your own drawings.

### Included models

Currently, the repository includes:

```text
models/
└── Roman_numeral_model.json
```

Additional models may be added as the project evolves.

> **Important:** These models are not general-purpose digit recognition models. They were trained using examples created for this project, so their performance depends heavily on the training data and handwriting styles they have seen.

---

## How the recognition pipeline works

### 1. Drawing

You draw a digit on a **280×280 canvas**.

### 2. Preprocessing

The drawing is processed before entering the neural network:

1. Convert the drawing to grayscale ink.
2. Detect the bounding box.
3. Crop the digit.
4. Scale it to approximately 20 pixels.
5. Center it using the center of mass.
6. Normalize the pixel values.
7. Convert the result into a **28×28 image**.

The final image contains:

**28 × 28 = 784 input values**

Each value represents the intensity of a pixel between **0 and 1**.

### 3. Forward propagation

The neural network processes the 784 inputs through the following architecture:

```text
784 → 64 → 32 → 9
```

The hidden layers use **ReLU** activation.

The output layer uses **softmax**, producing nine probabilities — one for each possible digit.

### 4. Training

During training, the network uses:

* Cross-entropy loss
* Backpropagation
* Gradient descent
* Mini-batches

The weights are updated after each training step to reduce the prediction error.

### 5. Prediction

The digit with the highest output probability becomes the network's prediction.

**Architecture:** `784 → 64 → 32 → 9`
**Parameters:** approximately **52,617**

---

## Technologies

* **HTML5 + CSS3**

  * CSS custom properties
  * Responsive grid
  * `prefers-color-scheme`
* **Vanilla JavaScript**

  * No frameworks
  * No bundlers
* **Canvas API**

  * Drawing
  * Image preprocessing
  * Neural network visualization
* **LocalStorage**

  * Training samples
  * Model weights
  * Training history
* **FileReader + Blob**

  * Model export/import
* **Float32Array**

  * Neural network weights
  * Numerical calculations

The neural network itself is implemented from scratch in plain JavaScript.

---

## Project structure

```text
Neural-Number-Lab/
│
├── index.html
├── style.css
├── script.js
│
├── models/
│   └── Roman_numeral_model.json
│
└── README.md
```

The `models/` directory contains exported models that can be loaded directly into the application.

---

## How to run

Clone the repository:

```bash
git clone https://github.com/LucaFilippi/Neural-Number-Lab.git
```

Then open:

```text
index.html
```

in any modern browser.

That's it.

**No build step.
No dependencies.
No server required.**

---

## Requirements

A modern browser with support for:

* Canvas API
* Pointer Events
* ES2020+

Supported browsers include:

* Chrome
* Firefox
* Edge
* Safari

---

## Training from scratch vs. using a pre-trained model

| Mode                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Train from scratch** | Start with a new model and teach it using your own drawings. |
| **Pre-trained model**  | Load a model from `models/` and start testing immediately.   |
| **Continue training**  | Load a pre-trained model and teach it additional examples.   |
| **Export model**       | Save the current network as a JSON file for later use.       |

Training from scratch is useful if your goal is to understand the learning process.

Using a pre-trained model allows you to immediately experiment with predictions, testing, and additional training.

---

## Known limitations

* Few training examples can lead to poor results.
* The network can memorize the examples it sees instead of learning general patterns.
* Variety is often more important than simply increasing the number of examples.
* Similar drawings can lead to narrow recognition.
* Training accuracy does not necessarily represent real-world accuracy.
* Always validate the model using new drawings in the **TEST** tab.
* Unbalanced datasets can bias predictions. For example, training with many `1`s and very few `7`s can make the network more likely to predict `1`.
* Performance depends heavily on the handwriting styles represented in the training data.
* The included pre-trained models are not intended to represent a universal handwriting dataset.

---

## Project goal

This project was built as a learning exercise to:

* Understand how an AI system can be built end-to-end, from raw pixels to predictions.
* Learn how neural networks work without relying on machine learning frameworks.
* Implement forward propagation, backpropagation, and gradient descent from scratch.
* Experiment with pattern recognition.
* Understand the importance of training data quality.
* Observe how hyperparameters affect training.
* Explore the difference between memorization and generalization.

The project is intentionally designed to be read, modified, broken, and rewritten.

The source code is organized to make the underlying concepts easier to explore.

---

## License

This project is released under the **MIT License**.

See the `LICENSE` file for the complete license text.

---

<p align="center">
  Made by <b>Luca Filippi</b>
</p>
