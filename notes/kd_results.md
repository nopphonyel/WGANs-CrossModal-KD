# Knowledge Distillation Results

อันนี้จะทำ KD แค่บน Generator เท่านั้นน้ะ

## Pretrained-Discriminator

| exp_code              | Arch Name         | Model size (MB) |  AlexFID  |  FID (InceptionNetV3)  | KD Algorithm        | Notes                                                       |
|:----------------------|-------------------|-----------------|:---------:|:----------------------:|---------------------|-------------------------------------------------------------|
| `kd.gen_kd.exp01`     | 3Layers-Generator | 0.763225        |  1507.04  |           -            | Discrim_T+AT+Logits | Too many constrain may not good.                            |
| `kd.gen_kd.exp01_1`   |                   |                 |  1921.13  |           -            | Discrim_T+Logits    |                                                             |
| `kd.gen_kd.exp01_2`   |                   |                 |  1650.81  |           -            | Discrim_T+AT        | Let's try using only each one layer for calculating AT loss |
| `kd.gen_kd.exp01_2_1` |                   |                 |           |                        | Discrim_T+1stAT     |                                                             |
| `kd.gen_kd.exp01_2_2` |                   |                 |           |                        | Discrim_T+2ndAT     |                                                             |
| `kd.gen_kd.exp01_2_3` |                   |                 |           |                        | Discrim_T+3rdAT     |                                                             |

**หมายเหตุ:**

- ช่องว่างในตารางหมายถึง เหมือนกับ Cell ด้านบน

**Some discussion**

- จากการทดลองมาได้สักพัก เป็นไปได้มั้ยที่ Discriminator มันเก่งเกินไปเลยทำให้ Generator ลู่เข้ายาก? เลยอยากจะทดลองใช้
  Discriminator ที่ไม่มีการ pre-trained