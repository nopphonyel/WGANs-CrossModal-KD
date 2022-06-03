# Classifier Results

งานนี้เราลอง fmri classifier อยู่หลาย arch ซึ่งขอสรุปดังนี้

| exp_code        | Arch name                                           | Max j2 acc  | Notes                |
|:----------------|:----------------------------------------------------|:-----------:|----------------------|
| `whole.exp01`   | SimpleFC 4 layers                                   |   62.50%    | Train แบบ whole      |
| `fe.exp01`      | SimpleFC 4 layers                                   |   56.20%    | Train แบบแยกเฉพาะ FE |
| `fe.exp01_1`    | SimpleFC 10 layers                                  |   54.17%    | Train แบบแยกเฉพาะ FE |
| `whole.exp02`   | Full-ResNet34                                       |   47.76%    | Train แบบ whole      |
| `whole.exp02_1` | Shallow-ResNet34 (No 3rd, 4th layer), Dropout = 0.2 |   46.45%    | Train แบบ whole      |
| `whole.exp02_2` | Shallow-ResNet34 (No 3rd, 4th layer)                |   50.23%    | Train แบบ whole      |
| `whole.exp03`   | Full-ResNet18                                       |   48.84%    | Train แบบ whole      |
