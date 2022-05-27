# Implementation States

Planned KD algorithm to implement or find

อธิบายเพิ่มเติมนิดหน่อย <br>
- `output` &rarr; Output จากโมเดลที่ผ่านทุก Layer แล้ว
- `l_i` &rarr; Output ที่ผ่านแต่ละ layer มาแล้ว 
  - เช่น `l_1` คือ output ที่ผ่าน layer 1 มาแล้ว

| Ready? | Name                                                          | Requirement             | More info?                                       | 
|:------:|:--------------------------------------------------------------|-------------------------|--------------------------------------------------|
| **Y**  | `Logits` (Regressing Logits)                                  | `output`                |                                                  |
| **Y**  | `ST` (Soft target)                                            | `output`                |                                                  |
| **Y**  | `AT` (Attention Transfer)                                     | `l_0`,`l_1`,`...`,`l_i` |                                                  |
|        | `AFD` (Attention Feature Distillation)                        |                         | [link](https://openreview.net/pdf?id=ryxyCeHtPB) |
|        | `CRD` (Contrastive Representation Distillation)               |                         |                                                  |
|        | `WCoRD` (Wasserstien Contrastive Representation Distillation) |                         |                                                  |
