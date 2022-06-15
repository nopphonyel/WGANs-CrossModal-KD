# To do list

On each day

## Pending...

Not that hurry task.

- [ ] อ่าน paper GCN อีกรอบนึง
- [ ] อ่าน
  paper [Compressing GANs using Knowledge Distillation](https://arxiv.org/pdf/1902.00159.pdf?ref=https://githubhelp.com)

## 2022-06-15

- [ ] Retrain GANs ทุกโมเดล แล้วใช้ FID Original ในการหาอันที่ดีที่สุดแทน -> **Training in progress...**
    - เท่าที่ดู เหมือนกับว่า FIDOrig ไม่ลดลงเลยหลังจาก epoch แรกไป
        - ถ้าอย่างนั้นทดลองใช้ DirectFID (คือเอารูปมาคำนวนเลย) แต่ว่า Metrics
          ตัวนี้ยังไม่เป็นที่ยอมรับเพราะเรามั่วขึ้นมาเอง
        - ใช้ PixelWise loss ก็ดูเข้าท่าเหมือนกัน เพราะเราไม่ได้ Generate รูป แต่เราทำ mapping จาก fmri -> stim_image
          ซึ่งมันมี stim image เป็น target อยู่แล้ว
- [ ] Retrain KD เช่นกัน

## 2022-06-04

- [x] ทำ script gen_kd ที่รวม teacher discriminator เข้าไปด้วย
    - อาจจะมีคำถามว่า ต้องจัด framework ยังไงบ้าง
        1. รวม Discriminator
        2. รวม Discriminator แล้ว train students เพิ่มอีกด้วย แต่ถ้าแบบนั้น มันจะไม่ได้เป็น offline มันเหมือนทำ online
           distillation มากกว่า

## 2022-06-03

- [x] จัดการที่อยู่ของ model ให้อยู่ที่เดียวกัน
- [x] ลอง run AlexNet for FID แล้วดูว่ายังโอเคมั้ย
    - ยังไม่แน่ใจอันนี้ อาจจะต้องลอง run full kd ก่อน
- [ ] ทำ gen_kd ที่รวม teacher Discriminator เข้าไปด้วย
    - [x] นั่นหมายความว่า เราต้อง export Discriminator ของ Teacher ออกมาด้วย
    - [x] ต้อง export ทุกโมเดลใน epoch เดียวกัน อย่าใช้คนละ epoch เพราะ parameter ที่ส่งต่อกัน อาจจะไม่รองรับกัน
      ทำให้ผลลัพธ์เพี้ยน

## 2022-06-02

- [x] ทำ AlexNet สำหรับ FID
- เนื่องจากว่า Olivier เหมือนจะรี Gourami หรือไม่ก็ไฟดับ

## 2022-06-01

- ไล่งานค้างให้เสร็จ

## 2022-05-31

- ตรวจสอบ FID
    - [x] ที่บอกว่า return Activation ออกมา มันอยู่ layer ไหนกันแน่?
        - อันนี้อาจจะไม่ต้องพยายามทำขนาดนั้น เพราะถ้า run จริง อาจจะใช้ pre-trained InceptionNet ไปเลย
- เขียน Script
    - [ ] Generator ที่มีการรวม Discriminator ด้วยตาม paper compressing GANs
    - [ ] Whole framework อาจจะต้องใช้ FIDOrig ด้วย เพราะ FID* มันอาจจะมั่วนิ่ม ไม่ถูกสุขลักษณะ
        - ตอนนี้ลองเปลี่ยนมา train AlexNet แบบไม่ยุ่งกับ loss ตัวอื่นๆเลย เพื่อมาทำ FID ของ Stimuli {B,R,I,A,N,S}
          โดยเฉพาะ

## 2022-05-30

- เขียน Script
    - [x] สำหรับ Whole framework แต่เพิ่ม FID และ save model ออกมาด้วย
- ลองไปหาดู
    - [x] GANs KD เพราะเรายังไม่แน่ใจว่าเวลา KD Generator เขาทำยังไงกัน
      -
      ได้เปเปอร์นี้มา: [Compressing GANs using Knowledge Distillation](https://arxiv.org/pdf/1902.00159.pdf?ref=https://githubhelp.com)

## 2022-05-29

- เขียน Script
    - [x] สำหรับ Train Classifier
        - [x] ลองรัน SimpleFC 10 layers
        - [x] Export SimpleFC 4 layers สำหรับ Generator-KD
    - [x] สำหรับ Train Generator ปกติ (ไม่ใช่ Generator1Block)
    - [x] ทำ Generator-KD
    - [x] FID Metrics
        - ตอนนี้เข้าใจล้ะว่าจะ implement ได้ยังไง
- Not so important
    - [x] ปรับให้ Reporter
        - ให้เก็บ Stack ของ extra message
        - เพิ่ม Time stamp ของแต่ละ extra message ด้วย