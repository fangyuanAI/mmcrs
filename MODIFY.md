Đây là **Problem Formulation + Model Architecture**

* ✅ Train được trực tiếp từ dataset Redial/Inspired
* ✅ Không giả định field ẩn

---

# 1. Problem Formulation

## 1.1 Notation

Cho một tập hội thoại:

[
\mathcal{D} = { (C^{(n)}, E^{(n)}, y^{(n)}) }_{n=1}^{N}
]

Trong đó:

* (C = {u_1, ..., u_T}): chuỗi utterance (`context`)
* (E = {e_1, ..., e_k}): tập entity trong hội thoại (`entity`)
* (y \in \mathcal{I}): item được recommend (`rec`)
* (\mathcal{I}): tập item

---

## 1.2 Conversational Recommendation (Standard)

Các mô hình như MSCRS học:

[
\max_\theta P(y \mid C, E)
]

Thông qua:

[
\hat{y} = f_\theta(C, E)
]

---

## 1.3 Confounding in Conversational Recommendation

Trong thực tế tồn tại popularity bias:

[
Z = popularity(y)
]

Cấu trúc phụ thuộc thực tế:

[
P \rightarrow C
]

[
P \rightarrow y
]

[
Z \rightarrow y
]

Do đó:

[
P(y \mid C, E) \neq P(y \mid do(P))
]

---

## 1.4 Reformulated Objective (CDH-CRS)

Ta định nghĩa latent preference representation:

[
z = g_\theta(C, E)
]

và tách:

[
z = z_{inv} + z_{spur}
]

Trong đó:

* (z_{inv}): preference invariant
* (z_{spur}): popularity-correlated component

Mục tiêu:

[
\max_\theta P(y \mid z_{inv})
]

đồng thời giảm phụ thuộc vào popularity.

---

# 2. Hypergraph-based Preference Modeling

## 2.1 Hypergraph Definition

Ta xây hypergraph:

[
\mathcal{H} = (V, \mathcal{E})
]

* (V = \mathcal{E}_{entity} \cup \mathcal{I})
* (\mathcal{E}): tập hyperedge

---

## 2.2 Hyperedge Construction

Với mỗi conversation n:

[
e^{(n)} =
{ e_1, e_2, ..., e_k }
]

Tức là:

> tất cả entity trong cùng một dialogue tạo thành một hyperedge.

Không cần thêm annotation.

---

## 2.3 Hypergraph Propagation

Cho:

* (X \in \mathbb{R}^{|V| \times d}): node embedding
* (H \in \mathbb{R}^{|V| \times |\mathcal{E}|}): incidence matrix

Update:

[
X' =
\sigma
\left(
D_v^{-1}
H
W_e
D_e^{-1}
H^T
X
W_v
\right)
]

Trong đó:

* (D_v): node degree matrix
* (D_e): hyperedge degree matrix

---

# 3. Model Architecture

## 3.1 Overview

```id="arch1"
Dialogue Encoder
        ↓
Entity Embedding Lookup
        ↓
Hypergraph Encoder
        ↓
Preference Aggregation
        ↓
Disentangled Projection
        ↓
Causal-aware Prediction
```

---

# 3.2 Dialogue Encoder

Encode context:

[
h_C = Encoder(C)
]

[
h_C \in \mathbb{R}^d
]

---

# 3.3 Entity-aware Hypergraph Encoder

Lấy embedding entity:

[
v_{e_i} = Emb(e_i)
]

Sau hypergraph propagation:

[
v_{e_i}' = HypergraphConv(v_{e_i})
]

Aggregate entity preference:

[
h_E = \frac{1}{k} \sum_{i=1}^{k} v_{e_i}'
]

---

# 3.4 Preference Representation

Kết hợp dialogue và entity:

[
z = W_c h_C + W_e h_E
]

---

# 4. Disentangled Preference Learning

Projection:

[
z_{inv} = W_{inv} z
]

[
z_{spur} = W_{spur} z
]

Orthogonality constraint:

[
L_{orth} =
\left|
z_{inv}^T z_{spur}
\right|_2
]

---

# 5. Causal-aware Recommendation Head

Prediction chỉ dùng invariant component:

[
\hat{y} =
softmax( W_r z_{inv} )
]

---

# 6. Popularity Debiased Objective

Tính popularity:

[
pop(y) =
\frac{# interactions(y)}
{\max_j # interactions(j)}
]

Weighted recommendation loss:

[
L_{rec} =
---------

\sum
\frac{1}{pop(y)^\gamma}
\log P(y \mid z_{inv})
]

---

# 7. Invariant Regularization

Chia batch theo popularity quantile thành K nhóm.

Loss invariant:

[
L_{inv} =
\sum_{k=1}^{K}
|
\nabla_{w|z_{inv}}
L_k
|^2
]

---

# 8. Final Objective

[
L =
L_{rec}
+
\lambda_1 L_{orth}
+
\lambda_2 L_{inv}
]

---

# 9. Sự khác biệt cốt lõi so với MSCRS

| MSCRS                  | CDH-CRS                 |
| ---------------------- | ----------------------- |
| Pairwise graph         | Hypergraph              |
| Correlation prediction | Invariant prediction    |
| Entangled embedding    | Disentangled            |
| No debias              | Popularity intervention |

---

# 10. Conceptual Contribution

1. Preference modeling unit chuyển từ pairwise entity → compositional hyperedge.
2. Prediction dựa trên invariant representation.
3. Popularity bias được xử lý qua interventional reweighting.
4. Không cần thêm annotation ngoài dataset gốc.
