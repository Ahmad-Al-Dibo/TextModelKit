# Mini GPT - Transformer-gebaseerd Taalmodel

Een eenvoudige implementatie van een GPT (Generative Pre-trained Transformer) model voor tekstgeneratie. Dit project demonstreert de kerncomponenten van moderne AI-taalmodellen.

---

## 📋 Inhoudsopgave

1. [Overzicht](#overzicht)
2. [Architectuur](#architectuur)
3. [Componentenanalyse](#componentenanalyse)
4. [Formules & Wiskunde](#formules--wiskunde)
5. [Werkingsstroom](#werkingsstroom)
6. [Configuratie](#configuratie)
7. [Installatie & Gebruik](#installatie--gebruik)

---

## 🎯 Overzicht

**Mini GPT** is een vereenvoudigde implementatie van een Transformer-gebaseerd taalmodel dat:
- Tekst tokenizeert (woorden → getallen)
- Self-Attention gebruikt om woorden met elkaar te verbinden
- Een neurale netwerk traint om het volgende woord te voorspellen
- Nieuwe tekst genereert op basis van een starttekst

**Kernidee:** Het model leert tekstpatronen door te voorspellen welk woord volgende komt in een reeks.

---

## 🏗️ Architectuur

### Systeem Diagram

```
Ruwe Tekst (data.txt)
    ↓
[TOKENIZER] → Woordvoorraadbouw (top 50k woorden)
    ↓
[ENCODER] → Woorden → Getallen (indices)
    ↓
[DATASET] → Blokken van 8 tokens (input/target paren)
    ↓
[DATALOADER] → Mini-batches (64 samples)
    ↓
[MINI GPT MODEL]
    ├─→ Token Embedding (woordrepresentatie)
    ├─→ Position Embedding (positie in zin)
    ├─→ 2× GPT Blokken
    │   ├─→ Self-Attention (woorden vergelijken)
    │   └─→ Feed-Forward (neural network)
    └─→ Output Head (volgende woord voorspellen)
    ↓
[LOSS CALCULATION] → Hoe goed is voorspelling?
    ↓
[BACKPROPAGATION] → Gewichten aanpassen
    ↓
[REPEAT] → 20 epochs × dataset
```

---

## 🔧 Componentenanalyse

### 1. **TOKENIZER** (`create_tokenizer`)

**Doel:** Tekst omzetten naar een getal-gebaseerd woordenboek

**Proces:**
```python
text = "hello world hello"
↓ (split)
tokens = ["hello", "world", "hello"]
↓ (count frequency)
frequencies = {"hello": 2, "world": 1}
↓ (keep top 50k)
vocab = ["<UNK>", "hello", "world"]
↓ (create mapping)
stoi = {"<UNK>": 0, "hello": 1, "world": 2}  # string to index
itos = {0: "<UNK>", 1: "hello", 2: "world"}  # index to string
```

**Parameters:**
- `max_vocab = 50000`: Maximale woordenboekgrootte
- `<UNK>` token: Voor onbekende woorden

**Output:**
- `vocab`: Liste van alle woorden
- `stoi`: Woordenboek (woord → getal)
- `itos`: Omgekeerd woordenboek (getal → woord)

---

### 2. **ENCODER** (`encode_data`)

**Doel:** Lijst van woorden in getallen omzetten

**Proces:**
```python
tokens = ["hello", "world", "unknown"]
stoi = {"<UNK>": 0, "hello": 1, "world": 2}
↓
encoded = [1, 2, 0]  # "unknown" → 0 (UNK)
```

**Veiligheidsfunctie:** Woorden die niet in woordenboek → UNK token (index 0)

---

### 3. **TextDataset** Klasse

**Doel:** Gecodeerde data in training-paren omzetten

**Datastructuur:**
```python
encoded = [1, 5, 3, 7, 2, 4, 8, 9, ...]
block_size = 8

Paren genereren:
Paar 1: X = [1,5,3,7,2,4,8], Y = [5,3,7,2,4,8,9]
Paar 2: X = [5,3,7,2,4,8,9], Y = [3,7,2,4,8,9,...]
        ↑                      ↑
        Input (7 woorden)      Target (7 woorden)
        Volgende woord voorspellen
```

**__len__:** Retourneert aantal mogelijke paren = len(data) - block_size
**__getitem__:** Retourneert één paar (X, Y) als tensors

---

### 4. **SelfAttention** Klasse

**Doel:** Relaties tussen woorden in een zin bepalen

**Wiskundige Formule:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Waar:
- $Q$ (Query): "Welke informatie zoek ik?"
- $K$ (Key): "Wat ben ik?"
- $V$ (Value): "Mijn inhoud/boodschap"
- $d_k$ = embedding dimensie (64)

**Proces stap voor stap:**

1. **Query, Key, Value berekenen:**
   ```
   Input X: shape (batch, time, embed_dim)
   Q = X @ W_Q
   K = X @ W_K
   V = X @ W_V
   ```

2. **Scores berekenen (compatibiliteit):**
   ```
   scores = Q @ K^T / sqrt(64)
   
   Betekenis: Hoeveel "aandacht" geeft query i naar key j?
   ```

3. **Masking (causal masking - alleen verleden zien):**
   ```
   Voor volgende woord voorspelling:
   Position 0: kan alleen position 0 zien
   Position 1: kan position 0,1 zien
   Position 2: kan position 0,1,2 zien
   ...
   (Toekomst is "onzichtbaar")
   
   mask = tril([[1,0,0], [1,1,0], [1,1,1]])
   ```

4. **Softmax (normalisatie):**
   ```
   weights = softmax(scores)
   Converteert scores naar kansen (0-1)
   ```

5. **Output berekenen:**
   ```
   output = weights @ V
   Gewogen combinatie van alle waarden
   ```

**Voorbeeld:**
```
Zin: "De kat eet"
Position 0 ("De"): 80% naar "kat", 20% naar zichzelf
Position 1 ("kat"): 10% naar "De", 70% naar "eet", 20% naar zichzelf
Position 2 ("eet"): 5% naar "De", 25% naar "kat", 70% naar zichzelf

Model leert welke woorden belangrijk zijn voor volgende woord!
```

---

### 5. **GPTBlock** Klasse

**Doel:** Volledige transformerblok (Self-Attention + Feed-Forward)

**Architectuur:**
```
Input X
  ↓
[LayerNorm] → Normalisatie
  ↓
[SelfAttention] → Woord-relaties
  ↓
[+ X] → Residual verbinding (skip connection)
  ↓ X_new
[LayerNorm] → Normalisatie
  ↓
[Feed-Forward: Linear(64→256) → ReLU → Linear(256→64)]
  ↓
[+ X_new] → Residual verbinding
  ↓
Output Y
```

**Residual verbinding:** `output = X + self_attn(norm(X)) + ff(norm(X_new))`

**Waarom?**
- Stabiliseert training (gradiënten verdwijnen niet)
- Informatie kan direct doorstromen

**Feed-Forward netwerk:**
```
Linear(64 → 256)  = Dimensie vergroten (detail toevoegen)
ReLU(x) = max(0, x)  = Non-lineariteit (besluitvorming)
Linear(256 → 64)  = Dimensie terugbrengen (samenvatting)
```

---

### 6. **MiniGPT** Klasse (Hoofd Model)

**Architectuur:**
```
Input: token_indices [batch, seq_len]
  ↓
Token Embedding: Woorden → Dense vectors (50000 → 64)
  ↓
Position Embedding: Posities → Dense vectors (8 → 64)
  ↓
[Combineer: token_emb + pos_emb]
  ↓
[GPTBlock 1] (Self-Attention + Feed-Forward)
  ↓
[GPTBlock 2] (Self-Attention + Feed-Forward)
  ↓
LayerNorm: Finale normalisatie
  ↓
Output Head: Transformeer naar woordvoorraadbkansen (64 → 50000)
  ↓
Output: logits [batch, seq_len, vocab_size]
```

**Forward Proces:**

1. **Token Embedding:**
   ```
   Token index 145 → 64-dimensionale vector
   Deze vector bevat "betekenis" van woord
   Geleerd tijdens training!
   ```

2. **Position Embedding:**
   ```
   Positie 0 → 64-dimensionale vector
   Positie 1 → 64-dimensionale vector
   ...
   Positie 7 → 64-dimensionale vector
   
   Model weet waar woord staat in zin
   ```

3. **Combinatie:**
   ```
   x = token_emb + pos_emb
   Element-wise som: betekenis + positie
   ```

4. **Twee Transformer Blokken:**
   ```
   Blok 1: Begint patronen te herkennen
   Blok 2: Verfijnt en bouwt erop voort
   ```

5. **Output Logits:**
   ```
   Voor elke positie in sequentie:
   Vector van 50000 getallen (één per woord)
   Hogere getal = waarschijnlijker volgende woord
   ```

---

### 7. **Training Process** (`train_model`)

**Algoritme: Stochastic Gradient Descent met AdamW Optimizer**

```
For epoch in 1..20:
    For batch in dataset:
        1. Forward Pass:
           logits = model(input_tokens)  # Voorspellingen
        
        2. Loss Calculation (Cross-Entropy):
           loss = -Σ log(P(target_i))
           
           Misfit tussen voorspelde en echte woorden
        
        3. Backward Pass:
           loss.backward()  # Gradiënten berekenen
        
        4. Update Weights:
           optimizer.step()  # Gewichten aanpassen
        
        5. Reset Gradiënten:
           optimizer.zero_grad()
    
    Print: Epoch X Loss: Y.YYYY Time: Z.ZZs
```

**Loss Functie (Cross-Entropy):**

$$\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N} \log(P(y_i | x_i))$$

Betekenis: "Hoe waarschijnlijk is het echte woord volgens het model?"
- Lager = Beter (model is zeker over juiste antwoord)
- Hoger = Slechter (model twijfelt)

**Optimizer (AdamW):**
- **Adaptive**: Snapt welke gewichten belangrijk zijn
- **Momentum**: Onthouden vorige updates (snelheid)
- **Weight Decay**: Voorkomen overfitting

---

### 8. **Generation** (`generate`)

**Autoregressive Generation (woord voor woord)**

```
Start: "English is"
↓ Encode: [142, 76]
↓ Tokenize: [142, 76]

Loop 20 keer (max_new_tokens):
    1. Neem laatste 8 tokens: [142, 76, ?, ?, ?, ?, ?, ?]
    2. Model.forward() → logits (kansen voor volgende woord)
    3. Neem top probability token
    4. Append: [142, 76, 512, ...]
    5. Repeat...

Result: [142, 76, 512, 89, 203, ...]
↓ Decode: "English is beautiful language..."
```

**Sampling methoden:**
```
Greedy: Kies woord met hoogste waarschijnlijkheid
        → Repetitief, voorspelbaar

Multinomial: Willekeurig sample volgens kansen
            → Meer variatie, soms onzin

Temperature: Pas kansen aan
            T=0.5: Voorzichtiger
            T=1.0: Normaal
            T=2.0: Wilder
```

---

## 📐 Formules & Wiskunde

### 1. **Embedding Dimensie**
```
vocab_size = 50,000 woorden
embed_dim = 64 (vector per woord)

Voorbeeldwordvector voor "hello":
[0.234, -0.512, 0.891, 0.123, ... (64 waarden totaal)]
```

### 2. **Attention Formula**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Waar:
- $M$ = Causal mask (-∞ voor toekomst)
- $\sqrt{d_k}$ = $\sqrt{64} = 8$ (schaling factor)

### 3. **Softmax Function**

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Converteert scores naar waarschijnlijkheden (0-1)

### 4. **Cross-Entropy Loss**

$$L = -\frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{V} y_{ij} \log(\hat{y}_{ij})$$

Waar:
- $y_{ij}$ = 1 als j het echte woord, 0 anders (one-hot)
- $\hat{y}_{ij}$ = Voorspelde waarschijnlijkheid van woord j
- $V$ = Woordvoorraadbgrootte (50,000)

### 5. **ReLU Function**

$$\text{ReLU}(x) = \max(0, x)$$

Basis non-lineariteit voor neurale netwerken

### 6. **LayerNorm**

$$y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Normaliseert activaties (helpt stabiliteit en snelheid)

---

## 🔄 Werkingsstroom

### **Volledige Pipeline**

#### **Fase 1: Voorbereiding (< 1 minuut)**

```
1. Load Data
   File: train-00031-of-00080.txt (82M tokens)
   → Sample: Eerste 1M tokens
   
2. Tokenization
   Text → Split op spaties
   Count frequencies → Top 50k woorden
   Create mappings (stoi, itos)
   
   Vocab grootte: 50,001 (inclusief <UNK>)

3. Encoding
   [Woord, Tokens, werden, geconverteerd]
   → [142, 523, 89, 401]
   
4. Dataset Creation
   Encoded data → Sliding window van 8 tokens
   
   Input:  [token0, token1, ..., token7]
   Target: [token1, token2, ..., token8]
   
5. DataLoader
   Dataset → Mini-batches van 64
   Random shuffle → Betere training
```

#### **Fase 2: Model Initialisatie (< 1 seconde)**

```
1. Create MiniGPT
   - Token Embeddings: 50001 × 64
   - Position Embeddings: 8 × 64
   - 2× GPT Blocks
   - Output Head: 64 → 50001
   
   Total Parameters: ~650k parameters

2. Optimizer
   AdamW learning_rate=0.001
   
3. Loss Function
   CrossEntropyLoss (voor classificatie)
```

#### **Fase 3: Training (10-20 minuten)**

```
Epoch 1/20:
  Batch 1: Loss = 10.8234
  Batch 2: Loss = 9.5412
  ...
  Total Loss: 9.2341 | Time: 35.42s

Epoch 2/20:
  Batch 1: Loss = 9.1234
  ...
  Total Loss: 8.7654 | Time: 34.89s

...

✓ Training completed in 0h 15m 42s
```

**Per Epoch:**
- ~16,000 batches (1M tokens / 64 batch_size)
- ~30-60 seconden op CPU
- Loss daalt geleidelijk (model wordt beter)

#### **Fase 4: Model Opslaan (1-5 seconden)**

```
Checkpoint bestand: mini_gpt.pth
Bevat:
  - Model gewichten (state_dict)
  - Optimizer status
  - Vocab/stoi/itos
  - Configuratie

Grootte: ~50-100 MB
```

#### **Fase 5: Text Generatie (< 1 seconde)**

```
Input: "English is"
↓
Encode: [142, 76]
↓
Loop 20 keer:
  Last 8 tokens: [142, 76, 512, 89, 203, 401, 145, 77]
  ↓
  Model forward → Logits
  ↓
  Softmax → Probabilities
  ↓
  Sample token: 234
  ↓
  Append: [142, 76, 512, 89, 203, 401, 145, 77, 234]
↓
Decode: "English is beautiful language of..."

Time: 0.23s
```

---

## ⚙️ Configuratie

```python
CONFIG = {
    "embed_dim": 64,        # Dimensie woordvectors
    "block_size": 8,        # Context window (8 woorden zien)
    "batch_size": 64,       # Mini-batch grootte
    "epochs": 20,           # Herhalingen over dataset
    "learning_rate": 1e-3,  # Stapgrootte optimizer
    "num_blocks": 2,        # Aantal transformer blokken
    "model_path": "mini_gpt.pth",
    "data_path": "train-00031-of-00080.txt",
}
```

**Tuning Tips:**
- ↑ `embed_dim`: Grotere modellen, meer geheugen
- ↑ `epochs`: Langer trainen, betere resultaten (tot een grens)
- ↓ `learning_rate`: Stabielere training, langzamer convergentie
- ↑ `num_blocks`: Meer lagen, meer complexiteit

---

## 🚀 Installatie & Gebruik

### **Requirements**
```bash
pip install torch numpy
```

### **Training Starten**
```bash
python gpt_v1.py
```

**Output:**
```
Using device: cpu
Vocab size: 50001
Data length: 999999

Starting training...
Epoch 1/20: Loss = 10.2341 | Time = 38.42s
Epoch 2/20: Loss = 9.1234 | Time = 37.89s
...
✓ Training completed in 0h 12m 15s

Saving model...
Model saved to mini_gpt.pth

Generating sample text...
Generated: english is a beautiful language for communication and learning
Generation time: 0.23s

==================================================
Total execution time: 0h 12m 21s
==================================================
```

### **Model Herstarten**
```bash
rm mini_gpt.pth
python gpt_v1.py
```

---

## 📊 Model Statistieken

| Aspect | Waarde |
|--------|--------|
| **Vocab Size** | 50,001 |
| **Embedding Dim** | 64 |
| **Context Window** | 8 tokens |
| **Attention Heads** | 1 (per block) |
| **Transformer Layers** | 2 |
| **Total Parameters** | ~650,000 |
| **Batch Size** | 64 |
| **Training Epochs** | 20 |
| **Estimated Training Time (CPU)** | 12-20 minuten |
| **Model File Size** | 50-100 MB |

---

## 🎓 Leerinhoud

### Concepten Geleerd

1. **Tokenization**: Tekst → Getallen
2. **Embeddings**: Woorden als vectoren
3. **Self-Attention**: Woord-relaties bepalen
4. **Transformers**: Multi-laag attention architectuur
5. **Autoregressive Generation**: Woord-voor-woord generatie
6. **PyTorch**: Framework voor neurale netwerken

### Wiskundige Concepten

- Matrixvermenigvuldiging
- Softmax normalisatie
- Cross-entropy loss
- Gradiënt descent
- Backpropagation

---

## 🔍 Veelgestelde Vragen

**V: Waarom is het zo langzaam op CPU?**
A: CPUs zijn niet geoptimaliseerd voor matrix-bewerkingen. GPUs zijn 10-50x sneller.

**V: Kan ik grotere modellen trainen?**
A: Ja! Verhoog `embed_dim`, `num_blocks`, en data-grootte. Gebruik GPU voor snelheid.

**V: Hoe verbeter ik gegenereerde tekst?**
A: Meer training, groter model, betere data, langere context window.

**V: Wat betekent "Loss daalt"?**
A: Model wordt beter in voorspellingen. Lagere loss = betere model.

---

## 📚 Referenties

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- PyTorch Documentation: https://pytorch.org

---

**Versie:** 1.0  
**Datum:** 2026-06-08  
**Status:** ✓ Werkend
